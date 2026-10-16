"""A recurrent field whose connections form a state-dependent integral operator.

The only communication between different spatial sites is

    m(r) = A(h(r),c(r)) integral kappa(r-r') B(h(r'),c(r')) V(h(r'),c(r')) dmu(r')

on a periodic domain with normalized measure dmu = dr / |Omega|. On a uniform
grid the quadrature weight is 1/(H*W). Fourier multipliers parameterize kappa;
the FFT evaluates these connections efficiently, rather than supplying a
separate additive message branch. Source and receiver factors are recomputed
from the whole previous state at every synchronous update.

``radius=R`` instead restricts kappa to a disc in real space, kappa(d) = 0 for
|d| > R cells (integer offsets, self included; R=1 is the 4 neighbours plus
self, sqrt(2) the 3x3 square), evaluated as a direct periodic convolution.
After T updates a site sees at most T*R cells away. ``rank=r`` factorizes it
as K(d) = P_out diag(w(d)) P_in (C->r->C channel projections, r weights per
offset). Offsets are in cells, so R has no fixed physical scale across grids.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from .recurrent import SpectralConv2d, _RecurrentField2d, _integer


def disc_offsets(radius):
    """Integer offsets d with |d| <= radius, as an (n, 2) tensor (self included)."""
    if isinstance(radius, bool) or not radius >= 1:
        raise ValueError("radius must be >= 1 cell")
    r = int(radius + 1e-9)
    axis = torch.arange(-r, r + 1)
    grid = torch.cartesian_prod(axis, axis)
    return grid[grid.square().sum(1) <= radius ** 2 + 1e-9]


class LocalConv2d(nn.Module):
    """Channel-mixing kernel supported on a disc, kappa(d) = 0 for |d| > radius cells; periodic grid."""

    def __init__(self, channels, radius, rank=None):
        super().__init__()
        self.channels = _integer("channels", channels)
        self.radius = float(radius)
        self.rank = None if rank is None else _integer("rank", rank)
        offsets = disc_offsets(radius)
        self.reach = int(offsets.max())
        self.register_buffer("offsets", offsets + self.reach, persistent=False)
        n = len(offsets)
        if self.rank is None:
            self.weight = nn.Parameter(torch.randn(channels, channels, n) / (channels * n) ** 0.5)
        else:
            r = self.rank
            self.proj_in = nn.Parameter(torch.randn(r, channels, 1, 1) / channels ** 0.5)
            self.proj_out = nn.Parameter(torch.randn(channels, r, 1, 1) / r ** 0.5)
            self.weight = nn.Parameter(torch.randn(r, 1, n) / n ** 0.5)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected (B,{self.channels},H,W) input")
        p = self.reach
        if min(x.shape[-2:]) < 2 * p + 1:
            raise ValueError(f"grid {tuple(x.shape[-2:])} too small for radius {self.radius}")
        size = 2 * p + 1
        kernel = self.weight.new_zeros(*self.weight.shape[:2], size, size)
        iy, ix = self.offsets.unbind(1)
        kernel[:, :, iy, ix] = self.weight
        x = F.pad(x, (p, p, p, p), mode="circular")
        if self.rank is None:
            return F.conv2d(x, kernel.to(x.dtype))
        x = F.conv2d(x, self.proj_in.to(x.dtype))
        x = F.conv2d(x, kernel.to(x.dtype), groups=self.rank)
        return F.conv2d(x, self.proj_out.to(x.dtype))


GATE_MODES = ("full", "frozen", "none", "source", "receiver", "static")


class StateDependentIntegral2d(nn.Module):
    """Factorized kernel K_ij = diag(A_i) kappa(r_i-r_j) diag(B_j).

    A and B are channel-wise gates in (0,2), not full pair-specific matrices.
    kappa mixes channels and may be signed/asymmetric; this is neither a
    probability attention matrix nor a guaranteed reciprocal interaction.
    The factorization avoids materializing a quadratic all-pairs edge tensor.

    ``gate_mode`` (ablations; the transmitted value V always uses the current state):
    ``full`` gates from [h^t, c] at every update; ``frozen`` gates from the initial
    state [h^0, c] = [tanh(c), c], i.e. computed once and held over the rollout;
    ``none`` A = B = 1; ``source`` / ``receiver`` only that gate (state-dependent),
    the other fixed at 1; ``static`` learned per-channel constants.
    """

    def __init__(self, width, modes, radius=None, rank=None, gate_mode="full"):
        super().__init__()
        self.width = _integer("width", width)
        if gate_mode not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {GATE_MODES}")
        self.gate_mode = gate_mode
        if gate_mode in ("full", "frozen", "source", "receiver"):
            self.gates = nn.Conv2d(2 * width, 2 * width, 1)
        elif gate_mode == "static":
            self.static_logits = nn.Parameter(torch.zeros(2 * width))
        self.value = nn.Conv2d(2 * width, width, 1, bias=False)
        self.kernel = SpectralConv2d(width, width, modes) if radius is None else LocalConv2d(width, radius, rank)

    def factors(self, state, forcing):
        """Source gates, receiver gates, and transmitted values, all (B,C,H,W)."""
        if state.ndim != 4 or state.shape != forcing.shape or state.shape[1] != self.width:
            raise ValueError(f"state and forcing must have matching (B,{self.width},H,W) shapes")
        context = torch.cat([state, forcing], dim=1)
        mode = self.gate_mode
        if mode == "none":
            source = receiver = torch.ones_like(state)
        elif mode == "static":
            source, receiver = (2 * self.static_logits.sigmoid()).view(1, -1, 1, 1).expand(
                state.shape[0], -1, *state.shape[-2:]).chunk(2, dim=1)
        else:
            gate_input = torch.cat([forcing.tanh(), forcing], dim=1) if mode == "frozen" else context
            source, receiver = (2 * self.gates(gate_input).sigmoid()).chunk(2, dim=1)
            if mode == "source":
                receiver = torch.ones_like(receiver)
            elif mode == "receiver":
                source = torch.ones_like(source)
        return source, receiver, self.value(context)

    def forward(self, state, forcing):
        source, receiver, value = self.factors(state, forcing)
        # Equal FFT normalizations yield the normalized quadrature 1/(H*W);
        # do not multiply by another cell-area factor here.
        return receiver * self.kernel(source * value)


class IntegralUpdate(nn.Module):
    """One shared interaction law followed by a strictly pointwise state update."""

    def __init__(self, width, modes, step_size, radius=None, rank=None, gate_mode="full"):
        super().__init__()
        self.step_size = step_size
        self.interaction = StateDependentIntegral2d(width, modes, radius, rank, gate_mode)
        self.proposal = nn.Conv2d(3 * width, width, 1)

    def forward(self, state, forcing):
        message = self.interaction(state, forcing)
        proposal = self.proposal(torch.cat([state, message, forcing], dim=1)).tanh()
        return (1 - self.step_size) * state + self.step_size * proposal


class RhizomeOperator2d(_RecurrentField2d):
    """State-dependent recurrent neural integral operator, not a separate FNO.

    Every cell uses the same synchronous interaction/update rule. The spatial
    kernel is translation-invariant, but the effective connections need not be:
    learned source/receiver factors depend on each site's current state and
    conditioning. There is no parallel local-convolution or spectral branch.

    Inputs: cond (B,in_channels,H,W), scalars (B,scalar_dim) when scalar_dim>0.
    Returns raw decoded fields; apply sigmoid for x_HI. ``return_states=True``
    returns (output, [h0,...,hT]). ``n_steps`` can vary for tied models; untied
    models cannot exceed the configured count.

    The normalized periodic domain and retained modes must have the same
    physical interpretation across grids. No cell-index parameters or spatial
    stencils are used. This supports consistent quadrature, not a guarantee of
    resolution-transfer accuracy, convergence, or stochastic generation.
    ``radius`` (cells) swaps in the compact real-space kernel; ``modes`` is then unused.
    ``gate_mode`` selects a gate ablation (see StateDependentIntegral2d); "frozen"
    relies on the initial state being tanh(forcing), as here.
    """

    def __init__(self, in_channels, out_channels=1, scalar_dim=0, width=24, modes=6,
                 n_steps=6, step_size=0.5, untied=False, radius=None, rank=None, gate_mode="full"):
        super().__init__(
            in_channels, out_channels, scalar_dim, width, modes, n_steps, step_size, untied,
            partial(IntegralUpdate, radius=radius, rank=rank, gate_mode=gate_mode),
        )


def periodic_upsample(x, factor):
    """Bilinear upsampling by an integer factor on a periodic grid (cell-centred, wraps at the edges)."""
    x = F.pad(x, (1, 1, 1, 1), mode="circular")
    x = F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False)
    return x[..., factor:-factor, factor:-factor]


class RhizomeUNet2d(nn.Module):
    """Two-level rhizome: a coarse rollout for long-range context, then a full-resolution rollout.

    The conditioning is average-pooled by ``factor`` (cells stay aligned with the
    fine grid, so rolls by multiples of ``factor`` commute with the model), lifted
    and rolled out for ``coarse_steps`` tied updates with its own interaction law
    (spectral with ``coarse_modes``, or compact with ``coarse_radius`` cells of the
    coarse grid). The coarse state is upsampled (periodic bilinear) and merged
    pointwise into the fine forcing, which drives ``n_steps`` tied fine updates
    (``modes`` or ``radius``, as in RhizomeOperator2d). Decoded from the fine
    state. One down/up level, no other skip paths.
    """

    def __init__(self, in_channels, out_channels=1, scalar_dim=0, width=24, modes=6, n_steps=3, step_size=0.5,
                 radius=None, rank=None, factor=4, coarse_modes=4, coarse_steps=3, coarse_radius=None,
                 coarse_rank=None):
        super().__init__()
        self.in_channels = _integer("in_channels", in_channels)
        self.out_channels = _integer("out_channels", out_channels)
        self.scalar_dim = _integer("scalar_dim", scalar_dim, minimum=0)
        self.width = _integer("width", width)
        self.n_steps = _integer("n_steps", n_steps)
        self.coarse_steps = _integer("coarse_steps", coarse_steps)
        self.factor = _integer("factor", factor, minimum=2)
        if isinstance(step_size, bool) or not 0 < step_size <= 1:
            raise ValueError("step_size must be in (0, 1]")
        self.lift_coarse = nn.Conv2d(in_channels, width, 1)
        self.lift_fine = nn.Conv2d(in_channels, width, 1)
        self.scalar_coarse = nn.Linear(scalar_dim, width) if scalar_dim else None
        self.scalar_fine = nn.Linear(scalar_dim, width) if scalar_dim else None
        self.coarse = IntegralUpdate(width, coarse_modes, float(step_size), coarse_radius, coarse_rank)
        self.fine = IntegralUpdate(width, modes, float(step_size), radius, rank)
        self.merge = nn.Conv2d(2 * width, width, 1)
        self.decode = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_channels, 1))

    @staticmethod
    def _forcing(lift, embed, cond, scalars):
        forcing = lift(cond)
        return forcing if embed is None else forcing + embed(scalars)[:, :, None, None]

    def forward(self, cond, scalars=None, *, n_steps=None, coarse_steps=None, return_states=False):
        """``n_steps`` / ``coarse_steps`` override the configured fine / coarse rollout lengths (tied updates)."""
        fine_steps = self.n_steps if n_steps is None else _integer("n_steps", n_steps)
        coarse_steps = self.coarse_steps if coarse_steps is None else _integer("coarse_steps", coarse_steps)
        if cond.ndim != 4 or cond.shape[1] != self.in_channels or min(cond.shape) < 1:
            raise ValueError(f"expected nonempty (B,{self.in_channels},H,W) conditioning")
        if cond.dtype not in (torch.float32, torch.float64):
            raise ValueError("conditioning must use float32 or float64")
        h, w = cond.shape[-2:]
        if h % self.factor or w % self.factor:
            raise ValueError(f"grid {(h, w)} is not divisible by factor {self.factor}")
        if (scalars is None) != (self.scalar_dim == 0) or (
                scalars is not None and scalars.shape != (cond.shape[0], self.scalar_dim)):
            raise ValueError(f"expected scalars with shape {(cond.shape[0], self.scalar_dim)}")
        forcing_c = self._forcing(self.lift_coarse, self.scalar_coarse, F.avg_pool2d(cond, self.factor), scalars)
        coarse = forcing_c.tanh()
        states = [coarse] if return_states else None
        for _ in range(coarse_steps):
            coarse = self.coarse(coarse, forcing_c)
            if return_states:
                states.append(coarse)
        forcing_f = self.merge(torch.cat([self._forcing(self.lift_fine, self.scalar_fine, cond, scalars),
                                          periodic_upsample(coarse, self.factor)], dim=1))
        state = forcing_f.tanh()
        if return_states:
            states.append(state)
        for _ in range(fine_steps):
            state = self.fine(state, forcing_f)
            if return_states:
                states.append(state)
        output = self.decode(state)
        return (output, states) if return_states else output


def build_model(config):
    """RhizomeOperator2d, or RhizomeUNet2d for ``config["kind"] == "unet"``, from a saved model config."""
    config = dict(config)
    kind = config.pop("kind", "rhizome")
    if kind == "unet":
        return RhizomeUNet2d(**config)
    if kind == "rhizome":
        return RhizomeOperator2d(**config)
    raise ValueError(f"unknown model kind {kind!r}")
