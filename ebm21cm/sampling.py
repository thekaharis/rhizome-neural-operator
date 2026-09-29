"""Samplers: EDM Heun predictor with optional energy-based MALA corrector.

The predictor integrates the probability-flow ODE (Karras et al. 2022,
Algorithm 2, with optional stochastic churn). After each step to noise
level sigma_{i+1} > 0, ``mala_steps`` Metropolis-adjusted Langevin moves
target the model's own noised density p_sigma(x) ~ exp(-E(x; sigma)).
The Metropolis test is possible only because the model has an explicit
energy; a score-only model can run Langevin but cannot accept/reject.

Step sizes adapt per sample towards the classic 0.574 acceptance rate.
Adaptation makes each corrector phase only approximately reversible; it is
meant to repair predictor error, not to be a certified MCMC chain.
"""

from __future__ import annotations

import math

import torch


def karras_sigmas(n, sigma_min=0.002, sigma_max=10.0, rho=7.0):
    """n noise levels from sigma_max to sigma_min, then 0. CPU float64 (MPS has no float64)."""
    i = torch.arange(n, dtype=torch.float64)
    s = (sigma_max ** (1 / rho) + i / max(n - 1, 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([s, torch.zeros(1, dtype=torch.float64)])


def _rand(shape, generator, device):
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)


def mala(model, x, sigma, cond, scalars, n_steps, eps, generator=None, target_accept=0.574):
    """n_steps MALA moves at fixed sigma. ``eps`` is a (B,) step-size tensor."""
    b = x.shape[0]
    sig = torch.full((b,), float(sigma), device=x.device)
    s, e = model.score(x, sig, cond, scalars)
    accepted = torch.zeros(b, device=x.device)
    for _ in range(n_steps):
        ev = eps.reshape(-1, 1, 1, 1)
        xp = x + ev * s + (2 * ev).sqrt() * _rand(x.shape, generator, x.device)
        sp, ep = model.score(xp, sig, cond, scalars)
        fwd = (xp - x - ev * s).pow(2).flatten(1).sum(1) / (4 * eps)
        bwd = (x - xp - ev * sp).pow(2).flatten(1).sum(1) / (4 * eps)
        log_alpha = (e - ep) + (fwd - bwd)
        u = torch.rand(b, generator=generator).to(x.device)
        acc = u.log() < log_alpha
        x = torch.where(acc.reshape(-1, 1, 1, 1), xp, x)
        s = torch.where(acc.reshape(-1, 1, 1, 1), sp, s)
        e = torch.where(acc, ep, e)
        accepted += acc.float()
        eps = eps * torch.where(acc, torch.tensor(1.0 + (1 - target_accept) * 0.2, device=x.device),
                                torch.tensor(1.0 - target_accept * 0.2, device=x.device))
    return x, eps, accepted / max(n_steps, 1)


@torch.no_grad()
def sample(model, cond, scalars, n_steps=32, sigma_min=0.002, sigma_max=10.0, rho=7.0,
           churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0,
           mala_steps=0, mala_scale=0.05, mala_sigma_max=1.0, generator=None, x_init=None):
    """Draw one sample per conditioning row. Returns (x, info)."""
    device = cond.device
    b, _, H, W = cond.shape
    shape = (b, model.n_target, H, W)
    t = karras_sigmas(n_steps, sigma_min, sigma_max, rho)
    x = (_rand(shape, generator, device) * t[0].float()) if x_init is None else x_init.clone()
    gamma_max = min(churn / n_steps, math.sqrt(2) - 1) if churn > 0 else 0.0
    accept_log, eps = [], None
    for i in range(n_steps):
        t_cur, t_next = float(t[i]), float(t[i + 1])
        gamma = gamma_max if s_tmin <= t_cur <= s_tmax else 0.0
        t_hat = t_cur + gamma * t_cur
        if gamma > 0:
            x = x + math.sqrt(t_hat ** 2 - t_cur ** 2) * s_noise * _rand(shape, generator, device)
        sig = torch.full((b,), t_hat, device=device)
        d, _ = model.denoise(x, sig, cond, scalars)
        dx = (x - d) / t_hat
        x_next = x + (t_next - t_hat) * dx
        if t_next > 0:
            sig2 = torch.full((b,), t_next, device=device)
            d2, _ = model.denoise(x_next, sig2, cond, scalars)
            x_next = x + (t_next - t_hat) * 0.5 * (dx + (x_next - d2) / t_next)
            if mala_steps and t_next <= mala_sigma_max:
                if eps is None:
                    eps = torch.full((b,), (mala_scale * t_next) ** 2, device=device)
                else:  # rescale the adapted step with the noise level
                    eps = eps * (t_next / t_cur) ** 2
                x_next, eps, rate = mala(model, x_next, t_next, cond, scalars, mala_steps, eps, generator)
                accept_log.append((t_next, float(rate.mean())))
        x = x_next
    return x, {"sigmas": t.tolist(), "mala_accept": accept_log}
