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

``pathways=R`` (MultiPathIntegral2d) gives R such terms, each with its own gates,
values and Fourier kernel (``kernel_groups`` dense channel blocks; = width is
depthwise), concatenated into the pointwise update rather than summed.
``disc_pathways`` of them can instead use a depthwise disc kernel of
``disc_radius`` cells applied by FFT (DiscFFTConv2d), whose cost does not grow
with the radius.
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


class GroupedSpectralConv2d(nn.Module):
    """SpectralConv2d with block-diagonal channel mixing: ``groups`` independent dense blocks.

    ``groups=1`` is SpectralConv2d (same retained modes and Hermitian handling);
    ``groups=channels`` is depthwise, one global Fourier filter per channel and no
    channel mixing. Input and output have the same number of channels.
    """

    def __init__(self, channels, modes, groups=1):
        super().__init__()
        self.channels = _integer("channels", channels)
        self.modes = _integer("modes", modes)
        self.groups = _integer("groups", groups)
        if channels % groups:
            raise ValueError(f"channels {channels} not divisible by groups {groups}")
        g, k = groups, channels // groups
        scale = (k * channels) ** -0.5  # (C*C)^-1/2 for dense, as SpectralConv2d; C^-1/2 for depthwise
        self.dc = nn.Parameter(scale * torch.randn(g, k, k))
        self.axis = nn.Parameter(scale * torch.randn(g, k, k, modes - 1, dtype=torch.cfloat))
        self.positive = nn.Parameter(scale * torch.randn(g, k, k, modes, modes - 1, dtype=torch.cfloat))
        self.negative = nn.Parameter(scale * torch.randn(g, k, k, modes - 1, modes - 1, dtype=torch.cfloat))

    def _mix(self, spectrum, weight):
        """(B,G,k,...) x (G,k,k,...) -> (B,G,k,...); elementwise when depthwise."""
        weight = weight.to(spectrum.dtype)
        if weight.shape[1] == 1:
            return spectrum * weight[:, 0]
        return torch.einsum("bgi...,gio...->bgo...", spectrum, weight)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected (B,{self.channels},H,W) input")
        h, w = x.shape[-2:]
        if min(h, w) < 2 * self.modes:
            raise ValueError(f"both grid dimensions must be >= 2*modes={2 * self.modes}, got {(h, w)}")
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        if x.dtype not in (torch.float32, torch.float64):
            raise ValueError("Fourier inputs must use float32 or float64")
        b, m = x.shape[0], self.modes
        spectrum = torch.fft.rfft2(x, norm="ortho").unflatten(1, (self.groups, -1))
        out = spectrum.new_zeros(spectrum.shape)
        out[..., 0, 0] = self._mix(spectrum[..., 0, 0].real, self.dc)
        if m > 1:
            axis = self._mix(spectrum[..., 1:m, 0], self.axis)
            out[..., 1:m, 0] = axis
            out[..., -(m - 1):, 0] = axis.flip(-1).conj()
            out[..., :m, 1:m] = self._mix(spectrum[..., :m, 1:m], self.positive)
            out[..., -(m - 1):, 1:m] = self._mix(spectrum[..., -(m - 1):, 1:m], self.negative)
        return torch.fft.irfft2(out.reshape(b, self.channels, h, w // 2 + 1), s=(h, w), norm="ortho")

    def filters(self):
        """Every retained coefficient of each (group, in, out) filter, as a (n_filters, n_coeffs) complex matrix."""
        parts = (self.dc.to(self.axis.dtype)[..., None], self.axis, self.positive.flatten(-2), self.negative.flatten(-2))
        return torch.cat(parts, dim=-1).flatten(0, 2)


class DiscFFTConv2d(nn.Module):
    """Depthwise kernel supported on a disc, kappa(d) = 0 for |d| > radius cells, evaluated by FFT.

    Same operator as a depthwise LocalConv2d, y(r) = sum_{|d|<=R} w(d) x(r+d) per
    channel on the periodic grid, but the weights are scattered into a full-grid
    kernel image and applied by multiplication over every Fourier mode. The cost
    does not depend on the radius. Offsets are in cells, as for LocalConv2d.
    """

    def __init__(self, channels, radius):
        super().__init__()
        self.channels = _integer("channels", channels)
        self.radius = float(radius)
        offsets = disc_offsets(radius)
        self.reach = int(offsets.max())
        self.register_buffer("offsets", offsets, persistent=False)
        self.weight = nn.Parameter(torch.randn(channels, len(offsets)) / len(offsets) ** 0.5)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected (B,{self.channels},H,W) input")
        h, w = x.shape[-2:]
        if min(h, w) < 2 * self.reach + 1:
            raise ValueError(f"grid {(h, w)} too small for radius {self.radius}")
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        # Correlation y(r) = sum_d w(d) x(r+d) is the convolution with kappa(-d) = w(d).
        kernel = x.new_zeros(self.channels, h, w)
        kernel[:, (-self.offsets[:, 0]) % h, (-self.offsets[:, 1]) % w] = self.weight.to(x.dtype)
        spectrum = torch.fft.rfft2(x) * torch.fft.rfft2(kernel)
        return torch.fft.irfft2(spectrum, s=(h, w))


PATHWAY_SHARING = ("", "gates", "kernels")


class MultiPathIntegral2d(nn.Module):
    """R communication pathways m_r = A_r * kappa_r(B_r * V_r), returned concatenated as (B, R*C, H, W).

    Each pathway has its own endpoint gates A_r, B_r in (0,2), its own transmitted
    values V_r (pointwise from [h, c]) and its own Fourier kernel kappa_r with
    ``groups`` dense channel blocks (1: dense C x C, C: depthwise). The pathways
    are concatenated, not summed; the update's pointwise proposal is the signed
    output mixing W_o (a separate linear W_o in front of it would be redundant).
    ``share="gates"``: one (A, B) for all pathways; ``share="kernels"``: one kappa.

    With nothing shared, R pathways of C channels are exactly one pathway of R*C
    channels with R*groups blocks: R widens the message, not the hidden state.
    pathways=1, groups=1 is StateDependentIntegral2d with gate_mode="full".

    ``disc_pathways=D`` makes the last D pathways use a depthwise disc kernel of
    ``disc_radius`` cells (DiscFFTConv2d: compact in space, every Fourier mode)
    instead of the Fourier one (global, ``modes`` lowest modes only).
    """

    def __init__(self, width, modes, pathways=1, groups=1, share="", disc_radius=None, disc_pathways=0):
        super().__init__()
        self.width = _integer("width", width)
        self.pathways = _integer("pathways", pathways)
        if share not in PATHWAY_SHARING:
            raise ValueError(f"share must be one of {PATHWAY_SHARING}")
        self.share = share
        self.disc_pathways = _integer("disc_pathways", disc_pathways, minimum=0)
        if disc_pathways > pathways or (disc_pathways > 0) != (disc_radius is not None):
            raise ValueError("disc_pathways must be in [1, pathways] exactly when disc_radius is given")
        if disc_pathways and share == "kernels":
            raise ValueError("share='kernels' needs every pathway to use the same kernel type")
        self.gates = nn.Conv2d(2 * width, 2 * width * (1 if share == "gates" else pathways), 1)
        self.value = nn.Conv2d(2 * width, width * pathways, 1, bias=False)
        spectral = pathways - disc_pathways
        if share == "kernels":
            self.kernel = GroupedSpectralConv2d(width, modes, groups)
        elif spectral:
            self.kernel = GroupedSpectralConv2d(width * spectral, modes, groups * spectral)
        else:
            self.kernel = None
        self.disc = DiscFFTConv2d(width * disc_pathways, disc_radius) if disc_pathways else None

    def factors(self, state, forcing):
        """Source gates, receiver gates, and transmitted values, all (B,R*C,H,W), pathway-major."""
        if state.ndim != 4 or state.shape != forcing.shape or state.shape[1] != self.width:
            raise ValueError(f"state and forcing must have matching (B,{self.width},H,W) shapes")
        context = torch.cat([state, forcing], dim=1)
        source, receiver = (2 * self.gates(context).sigmoid()).chunk(2, dim=1)
        if self.share == "gates":
            source, receiver = source.repeat(1, self.pathways, 1, 1), receiver.repeat(1, self.pathways, 1, 1)
        return source, receiver, self.value(context)

    def forward(self, state, forcing):
        source, receiver, value = self.factors(state, forcing)
        sent = source * value
        if self.share == "kernels":  # the same kappa on every pathway: fold pathways into the batch
            return receiver * self.kernel(sent.unflatten(1, (self.pathways, -1)).flatten(0, 1)).view_as(sent)
        if self.disc is None:
            return receiver * self.kernel(sent)
        if self.kernel is None:
            return receiver * self.disc(sent)
        split = (self.pathways - self.disc_pathways) * self.width
        return receiver * torch.cat([self.kernel(sent[:, :split]), self.disc(sent[:, split:])], dim=1)


def kernel_overlap(conv):
    """|cosine| Gram matrix of a GroupedSpectralConv2d's filters and its participation-ratio effective rank.

    Identical (collapsed) filters give |cos| = 1 off the diagonal and an effective rank near 1.
    """
    f = conv.filters().detach()
    f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gram = (f.conj() @ f.T).abs()
    eig = torch.linalg.eigvalsh(f.conj() @ f.T).clamp_min(0)
    return gram, float(eig.sum() ** 2 / eig.square().sum())


class IntegralUpdate(nn.Module):
    """One shared interaction law followed by a strictly pointwise state update.

    ``pathways`` / ``kernel_groups`` / ``share`` / ``disc_pathways`` other than
    (1, 1, "", 0) use MultiPathIntegral2d, whose R*C concatenated messages enter the proposal.
    """

    def __init__(self, width, modes, step_size, radius=None, rank=None, gate_mode="full",
                 pathways=1, kernel_groups=1, share="", disc_radius=None, disc_pathways=0):
        super().__init__()
        self.step_size = step_size
        if (pathways, kernel_groups, share, disc_pathways) == (1, 1, "", 0):
            self.interaction = StateDependentIntegral2d(width, modes, radius, rank, gate_mode)
        else:
            if radius is not None or gate_mode != "full":
                raise ValueError("multi-pathway interactions are spectral with gate_mode='full'")
            self.interaction = MultiPathIntegral2d(width, modes, pathways, kernel_groups, share,
                                                   disc_radius, disc_pathways)
        self.proposal = nn.Conv2d((2 + pathways) * width, width, 1)

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
    ``pathways`` / ``kernel_groups`` / ``share`` / ``disc_radius`` / ``disc_pathways``
    select MultiPathIntegral2d.
    """

    def __init__(self, in_channels, out_channels=1, scalar_dim=0, width=24, modes=6,
                 n_steps=6, step_size=0.5, untied=False, radius=None, rank=None, gate_mode="full",
                 pathways=1, kernel_groups=1, share="", disc_radius=None, disc_pathways=0):
        super().__init__(
            in_channels, out_channels, scalar_dim, width, modes, n_steps, step_size, untied,
            partial(IntegralUpdate, radius=radius, rank=rank, gate_mode=gate_mode,
                    pathways=pathways, kernel_groups=kernel_groups, share=share,
                    disc_radius=disc_radius, disc_pathways=disc_pathways),
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
