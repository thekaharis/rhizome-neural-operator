"""The rhizome operator on 3-D lightcone windows (transverse x, y; LOS z last).

Same interaction law as :mod:`ebm21cm.model.rhizome`, in three dimensions:

    m(r) = A(h, c)(r) * IFFT[ K * FFT(B(h, c) * V(h, c)) ](r)

with channel-wise source/receiver gates recomputed from the state at every
synchronous update, and all other maps pointwise. Differences from 2-D:

* **Non-periodic LOS.** Transverse planes are periodic box faces; the LOS is
  not. The FFT runs on the window zero-padded by ``los_pad`` cells along the
  LOS (zero sources transmit nothing). On the padded circle of length
  L = Z + los_pad, a source d cells ahead and one L-d cells behind share a
  kernel offset, so separations up to ``los_pad`` are wrap-free and larger ones
  alias; ``los_pad >= Z - 1`` gives an exactly linear (non-periodic) LOS
  convolution at twice the FFT length. Only the last axis is padded.
* **Anisotropic modes** ``(mx, my, mz)``: kx, ky in +-(m-1), kz in 0..mz-1
  (the LOS axis is the rfft axis). Both transverse dims must be >= 2*m and the
  padded LOS length >= 2*mz.
* **Factorized kernel (optional).** A dense C x C matrix per retained mode has
  ~4 m_x m_y m_z C^2 real parameters, hundreds of millions in 3-D. With
  ``rank=r`` the kernel is K(k) = P_out diag(w(k)) P_in: shared complex channel
  projections C->r->C and r complex weights per mode.
* **Memory.** ``checkpoint=True`` recomputes each update in the backward pass,
  so only the T+1 states of a rollout are stored.

Inputs follow fno-21cm's LOS-window interface: ``x`` is (B, C_in, X, Y, Z) with
the density, 1/(1+z), parameters, position and validity channels already
broadcast; the model lifts them pointwise. ``forward`` returns x_HI in [0, 1];
``logits=True`` returns pre-sigmoid values for BCE training.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .recurrent import _integer


class SpectralConv3d(nn.Module):
    """Low-mode 3-D Fourier kernel with zero-padded (non-periodic) last axis."""

    def __init__(self, channels, modes, los_pad=0, rank=None):
        super().__init__()
        self.channels = _integer("channels", channels)
        if len(modes) != 3:
            raise ValueError("modes must be (mx, my, mz)")
        self.modes = tuple(_integer("modes", m) for m in modes)
        self.los_pad = _integer("los_pad", los_pad, minimum=0)
        self.rank = None if rank is None else _integer("rank", rank)
        mx, my, mz = self.modes
        grid = (2 * mx - 1, 2 * my - 1, mz)
        if self.rank is None:
            scale = channels ** -1.0
            self.weight = nn.Parameter(scale * torch.randn(channels, channels, *grid, dtype=torch.cfloat))
        else:
            r = self.rank
            self.proj_in = nn.Parameter(channels ** -0.5 * torch.randn(channels, r, dtype=torch.cfloat))
            self.proj_out = nn.Parameter(r ** -0.5 * torch.randn(r, channels, dtype=torch.cfloat))
            self.weight = nn.Parameter(torch.randn(r, *grid, dtype=torch.cfloat) / (2 * mx * my * mz) ** 0.5)

    def _index(self, n, m):
        return torch.cat([torch.arange(m), torch.arange(n - (m - 1), n)])

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != self.channels:
            raise ValueError(f"expected (B,{self.channels},X,Y,Z) input")
        nx, ny, nz = x.shape[-3:]
        mx, my, mz = self.modes
        length = nz + self.los_pad
        if min(nx // 2, ny // 2) < max(mx, my) or length // 2 < mz:
            raise ValueError(f"grid {(nx, ny, length)} too small for modes {self.modes}")
        with torch.autocast(x.device.type, enabled=False):
            # FFTs stay in float32/float64 even under bf16 autocast.
            x = x if x.dtype == torch.float64 else x.float()
            if self.los_pad:
                x = F.pad(x, (0, self.los_pad))
            spec = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
            ix, iy = self._index(nx, mx).to(x.device), self._index(ny, my).to(x.device)
            low = spec[:, :, ix][:, :, :, iy][..., :mz]
            # Module.double() leaves complex parameters as complex64: cast here.
            cast = lambda w: w.to(spec.dtype)
            if self.rank is None:
                mixed = torch.einsum("bixyz,ioxyz->boxyz", low, cast(self.weight))
            else:
                mixed = torch.einsum("bixyz,ir->brxyz", low, cast(self.proj_in)) * cast(self.weight)
                mixed = torch.einsum("brxyz,ro->boxyz", mixed, cast(self.proj_out))
            out = spec.new_zeros(spec.shape[0], self.channels, *spec.shape[-3:])
            out[:, :, ix[:, None], iy[None, :], :mz] = mixed
            y = torch.fft.irfftn(out, s=(nx, ny, length), dim=(-3, -2, -1), norm="ortho")
        return y[..., :nz]


class IntegralUpdate3d(nn.Module):
    """Gated state-dependent integral interaction, then a pointwise convex update."""

    def __init__(self, width, modes, step_size, los_pad, rank):
        super().__init__()
        self.step_size = step_size
        self.gates = nn.Conv3d(2 * width, 2 * width, 1)
        self.value = nn.Conv3d(2 * width, width, 1, bias=False)
        self.kernel = SpectralConv3d(width, modes, los_pad, rank)
        self.proposal = nn.Conv3d(3 * width, width, 1)

    def forward(self, state, forcing):
        context = torch.cat([state, forcing], dim=1)
        source, receiver = (2 * self.gates(context).sigmoid()).chunk(2, dim=1)
        message = receiver * self.kernel(source * self.value(context)).to(source.dtype)
        proposal = self.proposal(torch.cat([state, message, forcing], dim=1)).tanh()
        return (1 - self.step_size) * state + self.step_size * proposal


class RhizomeOperator3d(nn.Module):
    """Tied (or untied) recurrent rhizome operator on (B, C_in, X, Y, Z) windows."""

    def __init__(self, in_channels, width=48, modes=(24, 24, 16), n_steps=6, step_size=0.5,
                 los_pad=64, rank=None, untied=False, checkpoint=True, amp=True):
        super().__init__()
        self.in_channels = _integer("in_channels", in_channels)
        self.width = _integer("width", width)
        self.n_steps = _integer("n_steps", n_steps)
        if isinstance(step_size, bool) or not 0 < step_size <= 1:
            raise ValueError("step_size must be in (0, 1]")
        self.untied, self.checkpoint, self.amp = bool(untied), bool(checkpoint), bool(amp)
        self.config = {"in_channels": in_channels, "width": width, "modes": list(modes), "n_steps": n_steps,
                       "step_size": step_size, "los_pad": los_pad, "rank": rank, "untied": untied,
                       "checkpoint": checkpoint, "amp": amp}
        self.lift = nn.Conv3d(in_channels, width, 1)
        self.cells = nn.ModuleList([IntegralUpdate3d(width, tuple(modes), float(step_size), los_pad, rank)
                                    for _ in range(n_steps if untied else 1)])
        self.decode = nn.Sequential(nn.Conv3d(width, width, 1), nn.GELU(), nn.Conv3d(width, 1, 1))

    def forward(self, x, *, logits=False):
        if x.ndim != 5 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected (B,{self.in_channels},X,Y,Z) windows")
        use_amp = self.amp and x.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            forcing = self.lift(x)
            state = forcing.tanh()
            for i in range(self.n_steps):
                cell = self.cells[i if self.untied else 0]
                if self.checkpoint and self.training and torch.is_grad_enabled():
                    state = grad_checkpoint(cell, state, forcing, use_reentrant=False)
                else:
                    state = cell(state, forcing)
            out = self.decode(state).float()
        return out if logits else out.sigmoid()
