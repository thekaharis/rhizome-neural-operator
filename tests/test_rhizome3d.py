"""3-D rhizome operator: shapes, symmetries, factorized kernel, checkpointing."""

import pytest
import torch

from ebm21cm.model.rhizome3d import RhizomeOperator3d, SpectralConv3d


def small(**kw):
    torch.manual_seed(0)
    return RhizomeOperator3d(4, width=6, modes=(3, 3, 2), n_steps=2, los_pad=4, amp=False, **kw).double()


def test_output_is_probability_with_window_shape():
    model = small()
    x = torch.randn(2, 4, 8, 8, 10, dtype=torch.float64)
    y = model(x)
    assert y.shape == (2, 1, 8, 8, 10) and 0 < y.min() and y.max() < 1
    assert torch.allclose(model(x, logits=True).sigmoid(), y)


def test_transverse_rolls_commute_los_does_not_wrap():
    model = small().eval()
    x = torch.randn(1, 4, 8, 8, 10, dtype=torch.float64)
    rolled = model(torch.roll(x, (3, 5), dims=(2, 3)))
    assert torch.allclose(rolled, torch.roll(model(x), (3, 5), dims=(2, 3)), atol=1e-10)
    # With LOS padding the operator is not periodic along the LOS.
    assert not torch.allclose(model(torch.roll(x, 3, dims=4)), torch.roll(model(x), 3, dims=4), atol=1e-6)


def test_full_los_padding_is_a_linear_convolution():
    torch.manual_seed(1)
    conv = SpectralConv3d(2, (3, 3, 4), los_pad=9).double()
    x = torch.zeros(1, 2, 8, 8, 10, dtype=torch.float64)
    x[..., -1] = torch.randn(1, 2, 8, 8, dtype=torch.float64)
    # A source in the last plane cannot reach earlier planes "from behind":
    # moving it one plane forward shifts the response exactly (no wrap).
    shifted = torch.zeros_like(x)
    shifted[..., -2] = x[..., -1]
    assert torch.allclose(conv(shifted)[..., :-1], conv(x)[..., 1:], atol=1e-10)


def test_factorized_kernel_is_smaller_and_trains():
    dense = SpectralConv3d(16, (6, 6, 4))
    low = SpectralConv3d(16, (6, 6, 4), rank=8)
    count = lambda m: sum(p.numel() * (2 if p.is_complex() else 1) for p in m.parameters())
    assert count(low) < count(dense) / 10
    model = small(rank=3)
    model(torch.randn(1, 4, 8, 8, 10, dtype=torch.float64), logits=True).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_checkpointing_matches_plain_gradients():
    x = torch.randn(1, 4, 8, 8, 10, dtype=torch.float64)
    grads = []
    for ckpt in (False, True):
        model = small(checkpoint=ckpt).train()
        model(x, logits=True).square().mean().backward()
        grads.append([p.grad.clone() for p in model.parameters()])
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-12)


def test_rejects_too_many_modes():
    with pytest.raises(ValueError):
        SpectralConv3d(2, (5, 3, 2))(torch.zeros(1, 2, 8, 8, 10))
