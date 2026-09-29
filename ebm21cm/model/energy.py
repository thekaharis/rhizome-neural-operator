"""Energy-parameterized EDM model.

The network defines an EDM-preconditioned field D(x; sigma) (Karras et al.
2022), and the model's *energy* is the scaled residual norm

    E(x; sigma | c) = || x - D(x; sigma | c) ||^2 / (2 sigma^2),

one scalar per sample, summed over pixels and both channels. The score is
its exact gradient, s = -grad_x E, so the learned vector field is
conservative by construction, and the effective denoiser used for training
and sampling is Tweedie's formula

    D_eff(x) = x + sigma^2 s(x).

Training matches D_eff to clean data with the EDM loss; this needs a second
differentiation through the network (double backward), roughly 2-3x the
cost of a plain score model. What that buys is a number: E can be compared
between candidate fields at the same (sigma, conditioning), used in
Metropolis corrections, and summed with other energies. For small
||Jacobian of D|| the gradient reduces to (x - D)/sigma^2, i.e. D is then
the denoiser itself. See Salimans & Ho (2021), "Should EBMs model the energy
or the score?", for this family of parameterizations.

Conditioning c = (density bands, standardized log(1+z) and parameters).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .unet import UNet


class EnergyEDM(nn.Module):
    def __init__(self, net: UNet, n_target: int, sigma_data: float):
        super().__init__()
        self.net = net
        self.n_target = n_target
        self.sigma_data = float(sigma_data)

    def _coeffs(self, sigma):
        sd = self.sigma_data
        s = sigma.reshape(-1, 1, 1, 1)
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * sd / (s ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (s ** 2 + sd ** 2).sqrt()
        c_noise = sigma.reshape(-1).log() / 4.0
        return s, c_skip, c_out, c_in, c_noise

    def field(self, x, sigma, cond, scalars):
        """The preconditioned network output D(x; sigma). Not the denoiser."""
        s, c_skip, c_out, c_in, c_noise = self._coeffs(sigma)
        f = self.net(torch.cat([c_in * x, cond], dim=1), c_noise, scalars)
        return c_skip * x + c_out * f

    def energy(self, x, sigma, cond, scalars):
        """(B,) energies. Differentiable in x and in the parameters."""
        r = x - self.field(x, sigma, cond, scalars)
        return 0.5 * r.pow(2).flatten(1).sum(1) / sigma.reshape(-1).pow(2)

    def score(self, x, sigma, cond, scalars, create_graph=False):
        """(-grad_x E, E). ``create_graph`` keeps the graph for training."""
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            e = self.energy(x, sigma, cond, scalars)
            (g,) = torch.autograd.grad(e.sum(), x, create_graph=create_graph)
        return -g, (e if create_graph else e.detach())

    def denoise(self, x, sigma, cond, scalars, create_graph=False):
        """Tweedie denoiser x + sigma^2 s(x), and the energy at x."""
        s, e = self.score(x, sigma, cond, scalars, create_graph=create_graph)
        d = x + sigma.reshape(-1, 1, 1, 1).pow(2) * s
        return (d if create_graph else d.detach()), e


def build_model(cfg: dict, n_cond: int, n_scalars: int, sigma_data: float, n_target: int = 2):
    net = UNet(in_ch=n_target + n_cond, out_ch=n_target, n_scalars=n_scalars,
               ch=cfg.get("ch", 64), mults=tuple(cfg.get("mults", (1, 2, 4))),
               num_res=cfg.get("num_res", 2), attn_levels=tuple(cfg.get("attn_levels", (2,))),
               heads=cfg.get("heads", 4), dropout=cfg.get("dropout", 0.0))
    return EnergyEDM(net, n_target, sigma_data)


def edm_loss(model: EnergyEDM, target, cond, scalars, p_mean=-1.2, p_std=1.2, generator=None):
    """EDM denoising loss on the energy's Tweedie denoiser. Returns (loss, sigma, per-sample)."""
    b = target.shape[0]
    rnd = torch.randn(b, device=target.device, generator=generator)
    sigma = (rnd * p_std + p_mean).exp()
    noise = torch.randn(target.shape, device=target.device, generator=generator)
    xt = target + noise * sigma.reshape(-1, 1, 1, 1)
    d, _ = model.denoise(xt, sigma, cond, scalars, create_graph=True)
    sd = model.sigma_data
    w = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2
    per = w * (d - target).pow(2).flatten(1).mean(1)
    return per.mean(), sigma, per
