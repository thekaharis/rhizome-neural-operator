import pytest
import torch

from ebm21cm.model import build_model
from ebm21cm.model.energy import edm_loss
from ebm21cm.sampling import karras_sigmas, mala, sample


def _randomize(model, seed=0):
    # Zero-initialized output layers make a fresh model trivially linear; perturb them.
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def _inputs(b=2, n_cond=3, n_scal=4, hw=16, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, 2, hw, hw, generator=g, dtype=dtype),
            torch.randn(b, n_cond, hw, hw, generator=g, dtype=dtype),
            torch.randn(b, n_scal, generator=g, dtype=dtype))


def test_score_is_minus_energy_gradient(tiny_model_cfg):
    model = _randomize(build_model(tiny_model_cfg, 3, 4, 0.7)).double().eval()
    x, c, s = _inputs(dtype=torch.float64)
    sig = torch.tensor([0.3, 2.0], dtype=torch.float64)
    score, e = model.score(x, sig, c, s)
    v = torch.randn(x.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    h = 1e-5
    fd = (model.energy(x + h * v, sig, c, s) - model.energy(x - h * v, sig, c, s)) / (2 * h)
    assert torch.allclose(fd, -(score * v).flatten(1).sum(1), rtol=1e-5, atol=1e-6)


def test_energies_are_per_sample(tiny_model_cfg):
    model = _randomize(build_model(tiny_model_cfg, 3, 4, 0.7)).eval()
    x, c, s = _inputs()
    sig = torch.tensor([0.5, 0.5])
    e0 = model.energy(x, sig, c, s)
    x2 = x.clone()
    x2[0] += 1.0
    e1 = model.energy(x2, sig, c, s)
    assert not torch.isclose(e0[0], e1[0]) and torch.isclose(e0[1], e1[1])


def test_roll_invariance(tiny_model_cfg):
    """Circular padding: rolling every input by a multiple of 2**(levels-1) leaves E unchanged."""
    model = _randomize(build_model(tiny_model_cfg, 3, 4, 0.7)).double().eval()
    x, c, s = _inputs(dtype=torch.float64)
    sig = torch.tensor([0.4, 1.3], dtype=torch.float64)
    e = model.energy(x, sig, c, s)
    roll = lambda a: torch.roll(a, shifts=(4, -6), dims=(-2, -1))
    assert torch.allclose(e, model.energy(roll(x), sig, roll(c), s), rtol=1e-10)


def test_fresh_model_is_identity_denoiser_limit(tiny_model_cfg):
    """With the zero-initialized output, D = c_skip x and the energy is analytic."""
    model = build_model(tiny_model_cfg, 3, 4, 0.5).double().eval()
    x, c, s = _inputs(dtype=torch.float64)
    sig = torch.tensor([0.2, 3.0], dtype=torch.float64)
    c_skip = 0.25 / (sig ** 2 + 0.25)
    expect = 0.5 * ((1 - c_skip) ** 2) * x.pow(2).flatten(1).sum(1) / sig ** 2
    assert torch.allclose(model.energy(x, sig, c, s), expect)


def test_loss_backprops_through_score(tiny_model_cfg):
    model = build_model(tiny_model_cfg, 3, 4, 0.7)
    x, c, s = _inputs()
    loss, sig, per = edm_loss(model, x, c, s)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and per.shape == (2,)
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_sampler_shapes_and_mala(tiny_model_cfg):
    model = _randomize(build_model(tiny_model_cfg, 3, 4, 0.7)).eval()
    _, c, s = _inputs()
    t = karras_sigmas(8, 0.01, 10.0)
    assert float(t[0]) == pytest.approx(10.0) and float(t[-2]) == pytest.approx(0.01) and t[-2] > 0 and t[-1] == 0 and bool((t[:-1].diff() < 0).all())
    g = torch.Generator().manual_seed(0)
    x, info = sample(model, c, s, n_steps=6, sigma_max=10.0, mala_steps=2, mala_sigma_max=5.0, generator=g)
    assert x.shape == (2, 2, 16, 16) and torch.isfinite(x).all()
    assert info["mala_accept"] and all(0.0 <= r <= 1.0 for _, r in info["mala_accept"])
    x0 = torch.randn(2, 2, 16, 16, generator=g)
    y, eps, rate = mala(model, x0, 0.5, c, s, 3, torch.full((2,), 1e-4), g)
    assert y.shape == x0.shape and eps.shape == (2,) and rate.shape == (2,)


def test_sampler_is_seeded(tiny_model_cfg):
    model = _randomize(build_model(tiny_model_cfg, 3, 4, 0.7)).eval()
    _, c, s = _inputs()
    a, _ = sample(model, c, s, n_steps=4, generator=torch.Generator().manual_seed(5))
    b, _ = sample(model, c, s, n_steps=4, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)
