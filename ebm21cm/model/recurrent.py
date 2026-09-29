"""Synchronous recurrent local–Fourier operators on periodic 2-D fields.

Cells exchange hidden states, not independently trained weights. One shared
update is unrolled in computational time, without any spatial traversal order.
An untied version uses independent copies of the update as a depth-matched
feed-forward baseline. Neither version defines a stochastic sampler.
"""

from __future__ import annotations

from numbers import Integral

import torch
from torch import nn


def _integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class SpectralConv2d(nn.Module):
    """Low-mode Fourier channel mixing, with signed transverse frequencies.

    Retains ky = -(modes-1), ..., modes-1 and kx = 0, ..., modes-1
    in the real FFT. Both grid dimensions must be >= 2*modes, keeping
    the retained bands separate and below the Nyquist frequency.
    Weights are reusable at other supported resolutions on the same domain;
    this alone does not guarantee physically accurate resolution transfer.
    """

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = _integer("in_channels", in_channels)
        self.out_channels = _integer("out_channels", out_channels)
        self.modes = _integer("modes", modes)
        scale = (in_channels * out_channels) ** -0.5
        # kx=0 is self-conjugate: real DC and one complex side for ky.
        # Learning the other side independently would introduce dead directions
        # that irfft2 silently projects away.
        self.dc = nn.Parameter(scale * torch.randn(in_channels, out_channels))
        self.axis = nn.Parameter(scale * torch.randn(
            in_channels, out_channels, modes - 1, dtype=torch.cfloat))
        # kx>0 permits independent signed ky bands.
        self.positive = nn.Parameter(scale * torch.randn(
            in_channels, out_channels, modes, modes - 1, dtype=torch.cfloat))
        self.negative = nn.Parameter(scale * torch.randn(
            in_channels, out_channels, modes - 1, modes - 1, dtype=torch.cfloat))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected (B,{self.in_channels},H,W) input")
        h, w = x.shape[-2:]
        if min(h, w) < 2 * self.modes:
            raise ValueError(f"both grid dimensions must be >= 2*modes={2 * self.modes}, got {(h, w)}")
        if x.dtype not in (torch.float32, torch.float64):
            raise ValueError("Fourier inputs must use float32 or float64")
        spectrum = torch.fft.rfft2(x, norm="ortho")
        out = spectrum.new_zeros(x.shape[0], self.out_channels, h, w // 2 + 1)
        m = self.modes
        out[:, :, 0, 0] = torch.einsum("bi,io->bo", spectrum[:, :, 0, 0].real, self.dc)
        # Module.double() does not convert complex parameters; cast here to
        # match the input spectrum while retaining gradients to the parameters.
        if m > 1:
            axis = torch.einsum("bik,iok->bok", spectrum[:, :, 1:m, 0], self.axis.to(spectrum.dtype))
            out[:, :, 1:m, 0] = axis
            out[:, :, -(m - 1):, 0] = axis.flip(-1).conj()
            out[:, :, :m, 1:m] = torch.einsum(
                "bihw,iohw->bohw", spectrum[:, :, :m, 1:m], self.positive.to(spectrum.dtype))
            out[:, :, -(m - 1):, 1:m] = torch.einsum(
                "bihw,iohw->bohw", spectrum[:, :, -(m - 1):, 1:m], self.negative.to(spectrum.dtype))
        return torch.fft.irfft2(out, s=(h, w), norm="ortho")


class LocalFourierUpdate(nn.Module):
    """A simultaneous update from a frozen previous state and fixed forcing."""

    def __init__(self, width, modes, step_size, use_local, use_spectral):
        super().__init__()
        self.step_size = step_size
        self.pointwise = nn.Conv2d(width, width, 1)
        self.local = nn.Conv2d(width, width, 3, padding=1, padding_mode="circular") if use_local else None
        self.spectral = SpectralConv2d(width, width, modes) if use_spectral else None

    def forward(self, state, forcing):
        messages = self.pointwise(state) + forcing
        if self.local is not None:
            messages = messages + self.local(state)
        if self.spectral is not None:
            messages = messages + self.spectral(state)
        return (1 - self.step_size) * state + self.step_size * messages.tanh()


class _RecurrentField2d(nn.Module):
    """Shared field initialization, synchronous rollout, and pointwise decoding.

    ``cond`` is (B,in_channels,H,W); optional ``scalars`` is (B,scalar_dim).
    Returns raw decoded fields/logits. For x_HI, the caller applies sigmoid.
    ``return_states=True`` returns (output, [h0, ..., hT]).

    For 0 < step_size <= 1 the hidden state is bounded by one because every
    update is a convex combination with tanh. This is NOT a contraction or
    convergence guarantee. Extra updates beyond training are diagnostics.

    With ``untied=True``, update blocks have independent parameters. Rollout
    may be shortened, but cannot exceed the configured number of blocks.
    No spatial pooling/coordinates are used: arbitrary toroidal rolls commute
    with the model (up to floating-point error).
    """

    def __init__(self, in_channels, out_channels, scalar_dim, width, modes,
                 n_steps, step_size, untied, make_update):
        super().__init__()
        self.in_channels = _integer("in_channels", in_channels)
        self.out_channels = _integer("out_channels", out_channels)
        self.scalar_dim = _integer("scalar_dim", scalar_dim, minimum=0)
        self.width = _integer("width", width)
        self.modes = _integer("modes", modes)
        self.n_steps = _integer("n_steps", n_steps)
        if isinstance(step_size, bool) or not 0 < step_size <= 1:
            raise ValueError("step_size must be in (0, 1]")
        self.untied = bool(untied)
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.scalar_embed = nn.Linear(scalar_dim, width) if scalar_dim else None
        self.cells = nn.ModuleList([
            make_update(width, modes, float(step_size))
            for _ in range(n_steps if untied else 1)
        ])
        self.decode = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_channels, 1))

    def forward(self, cond, scalars=None, *, n_steps=None, return_states=False):
        if cond.ndim != 4 or cond.shape[1] != self.in_channels or min(cond.shape) < 1:
            raise ValueError(f"expected nonempty (B,{self.in_channels},H,W) conditioning")
        if cond.dtype not in (torch.float32, torch.float64):
            raise ValueError("conditioning must use float32 or float64")
        steps = self.n_steps if n_steps is None else _integer("n_steps", n_steps)
        if self.untied and steps > self.n_steps:
            raise ValueError("untied rollout cannot exceed its configured number of update blocks")
        if scalars is None:
            if self.scalar_dim:
                raise ValueError(f"scalars with shape {(cond.shape[0], self.scalar_dim)} are required")
        elif scalars.shape != (cond.shape[0], self.scalar_dim):
            raise ValueError(f"expected scalars with shape {(cond.shape[0], self.scalar_dim)}")
        forcing = self.lift(cond)
        if self.scalar_embed is not None:
            forcing = forcing + self.scalar_embed(scalars)[:, :, None, None]
        state = forcing.tanh()
        states = [state] if return_states else None
        for i in range(steps):
            state = self.cells[i if self.untied else 0](state, forcing)
            if return_states:
                states.append(state)
        output = self.decode(state)
        return (output, states) if return_states else output


class RecurrentFNO2d(_RecurrentField2d):
    """Original additive local + Fourier baseline, with shared update weights.

    See ``_RecurrentField2d`` for input shapes and rollout behavior. Parameter
    names and initialization order are unchanged, so existing checkpoints load.
    """

    def __init__(self, in_channels, out_channels=1, scalar_dim=0, width=24, modes=6,
                 n_steps=6, step_size=0.5, untied=False, use_local=True, use_spectral=True):
        super().__init__(
            in_channels, out_channels, scalar_dim, width, modes, n_steps, step_size, untied,
            lambda w, m, dt: LocalFourierUpdate(w, m, dt, use_local, use_spectral),
        )
