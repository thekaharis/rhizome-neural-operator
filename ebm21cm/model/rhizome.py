"""A recurrent field whose connections form a state-dependent integral operator.

The only communication between different spatial sites is

    m(r) = A(h(r),c(r)) integral kappa(r-r') B(h(r'),c(r')) V(h(r'),c(r')) dmu(r')

on a periodic domain with normalized measure dmu = dr / |Omega|. On a uniform
grid the quadrature weight is 1/(H*W). Fourier multipliers parameterize kappa;
the FFT evaluates these connections efficiently, rather than supplying a
separate additive message branch. Source and receiver factors are recomputed
from the whole previous state at every synchronous update.
"""

from __future__ import annotations

import torch
from torch import nn

from .recurrent import SpectralConv2d, _RecurrentField2d, _integer


class StateDependentIntegral2d(nn.Module):
    """Factorized kernel K_ij = diag(A_i) kappa(r_i-r_j) diag(B_j).

    A and B are channel-wise gates in (0,2), not full pair-specific matrices.
    kappa mixes channels and may be signed/asymmetric; this is neither a
    probability attention matrix nor a guaranteed reciprocal interaction.
    The factorization avoids materializing a quadratic all-pairs edge tensor.
    """

    def __init__(self, width, modes):
        super().__init__()
        self.width = _integer("width", width)
        self.gates = nn.Conv2d(2 * width, 2 * width, 1)
        self.value = nn.Conv2d(2 * width, width, 1, bias=False)
        self.kernel = SpectralConv2d(width, width, modes)

    def factors(self, state, forcing):
        """Source gates, receiver gates, and transmitted values, all (B,C,H,W)."""
        if state.ndim != 4 or state.shape != forcing.shape or state.shape[1] != self.width:
            raise ValueError(f"state and forcing must have matching (B,{self.width},H,W) shapes")
        context = torch.cat([state, forcing], dim=1)
        source, receiver = (2 * self.gates(context).sigmoid()).chunk(2, dim=1)
        return source, receiver, self.value(context)

    def forward(self, state, forcing):
        source, receiver, value = self.factors(state, forcing)
        # Equal FFT normalizations yield the normalized quadrature 1/(H*W);
        # do not multiply by another cell-area factor here.
        return receiver * self.kernel(source * value)


class IntegralUpdate(nn.Module):
    """One shared interaction law followed by a strictly pointwise state update."""

    def __init__(self, width, modes, step_size):
        super().__init__()
        self.step_size = step_size
        self.interaction = StateDependentIntegral2d(width, modes)
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
    """

    def __init__(self, in_channels, out_channels=1, scalar_dim=0, width=24, modes=6,
                 n_steps=6, step_size=0.5, untied=False):
        super().__init__(
            in_channels, out_channels, scalar_dim, width, modes, n_steps, step_size, untied,
            IntegralUpdate,
        )
