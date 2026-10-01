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


def test_warm_start_signature_is_backward_compatible():
    import ebm21cm.train_rhizome3d as tr
    parser_args = ["--run-dir", "x"]
    # The parser exposes the warm-start options...
    import argparse, inspect
    source = inspect.getsource(tr.main)
    assert '"--init-from"' in source and '"--init-optimizer"' in source
    # ...and drops them from the resume signature when unused.
    assert 'signature.pop("init_from", None)' in source


STATS = {"density_offset": 0.0, "density_scale": 10.0, "velocity_offset": 0.0, "velocity_scale": 100.0,
         "tb_offset": -20.0, "tb_scale": 40.0, "omm_mean": 0.3, "omm_std": 0.05}
INDICES = {"density": 0, "velocity": 1, "z": 2, "omm": 12, "relative": 14}


def mf_inputs(batch=1, nz=10):
    torch.manual_seed(3)
    x = 0.1 * torch.randn(batch, 16, 8, 8, nz, dtype=torch.float64)
    x[:, 2] = 1 / (1 + torch.linspace(7, 9, nz, dtype=torch.float64))   # 1/(1+z) along the LOS
    x[:, 14] = (torch.arange(nz, dtype=torch.float64) * 1.43 / 1000)      # relative LOS position (Gpc)
    return x


def test_multifield_heads_shapes():
    x = mf_inputs()
    for head, extra in (("plain", {}), ("structured", {"structured": {"indices": INDICES, "stats": STATS}})):
        torch.manual_seed(0)
        model = RhizomeOperator3d(16, width=6, modes=(3, 3, 2), n_steps=2, los_pad=4, amp=False,
                                  targets=("neutral_fraction", "brightness_temp"), tb_head=head, **extra).double()
        y = model(x)
        assert y.shape == (1, 2, 8, 8, 10) and 0 < y[:, 0].min() and y[:, 0].max() < 1
        logits = model(x, logits=True)
        assert torch.allclose(logits[:, 0].sigmoid(), y[:, 0]) and torch.allclose(logits[:, 1], y[:, 1])


def test_structured_tb_vanishes_where_ionized():
    from ebm21cm.model.rhizome3d import StructuredBrightness3d
    head = StructuredBrightness3d(INDICES, STATS)
    x, u = mf_inputs().float(), torch.randn(1, 1, 8, 8, 10)
    tb = head(x, torch.zeros(1, 1, 8, 8, 10), u) * STATS["tb_scale"] + STATS["tb_offset"]
    assert torch.allclose(tb, torch.zeros_like(tb), atol=1e-4)


def test_structured_tb_matches_fno_head():
    import sys
    from pathlib import Path
    root = Path("/pfs/10/work/hd_id260-fno_training/fno-21cm")
    if not (root / "multifield_model.py").exists():
        pytest.skip("fno-21cm checkout not available")
    sys.path.insert(0, str(root))
    try:
        from multifield_model import StructuredBrightness
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot import fno-21cm: {exc}")
    from ebm21cm.model.rhizome3d import StructuredBrightness3d
    ours = StructuredBrightness3d(INDICES, STATS)
    fno = StructuredBrightness((0, 1, 2, 12, 14, 0, 1), STATS)
    x, u, xhi = mf_inputs().float(), torch.randn(1, 1, 8, 8, 10), torch.rand(1, 1, 8, 8, 10)
    assert torch.allclose(ours(x, xhi, u), fno(x, xhi, u).float(), atol=1e-5)


def test_xhi_only_model_is_unchanged_by_multifield_support():
    torch.manual_seed(0)
    model = RhizomeOperator3d(4, width=6, modes=(3, 3, 2), n_steps=2, los_pad=4, amp=False)
    assert model.decode[-1].out_channels == 1 and model.tb_head is None and model.structured is None
    assert all(not k.startswith("structured") for k in model.state_dict())


def test_multifield_config_validation():
    with pytest.raises(ValueError):
        RhizomeOperator3d(4, width=6, modes=(3, 3, 2), targets=("brightness_temp",))
    with pytest.raises(ValueError):
        RhizomeOperator3d(4, width=6, modes=(3, 3, 2), targets=("neutral_fraction", "brightness_temp"))
    with pytest.raises(ValueError):
        RhizomeOperator3d(4, width=6, modes=(3, 3, 2), targets=("neutral_fraction", "brightness_temp"),
                          tb_head="structured")
