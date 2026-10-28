import json

import h5py
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ebm21cm.cosmology import FlatLCDM
from ebm21cm.data.lightcone import LC_CELL_MPC, LC_Z_MIN, NODE_REDSHIFTS, node_weight
from ebm21cm.data.mf_slices import SCHEMA, MFMemorySplit
from ebm21cm.model.rhizome import build_model
from ebm21cm.train_rhizome_mf2d import MFConfig, evaluate, node_output, node_scalars, run, two_node_output


@pytest.mark.parametrize("mode", ["chi", "z"])
def test_node_weight_is_zero_at_node_a_and_tends_to_one_at_node_b(mode):
    zn = NODE_REDSHIFTS[(NODE_REDSHIFTS > 5) & (NODE_REDSHIFTS < 25)]
    assert np.all(node_weight(zn, 0.31, mode) == 0.0)
    near_b = node_weight(zn[:-1] + 1e-9, 0.31, mode)  # just above the lower node b of the interval below zn[1:]
    assert np.all(near_b < 1) and np.all(near_b > 1 - 1e-7)
    z = np.linspace(zn[3], zn[4], 50, endpoint=False)[1:]
    w = node_weight(z, 0.31, mode)
    assert np.all(np.diff(w) < 0) and np.all((w > 0) & (w < 1))


def test_chi_weight_matches_tabulated_distances_and_z_weight_to_first_order():
    z = np.linspace(5.0, 25.0, 4001)
    cosmo = FlatLCDM(0.27, 0.6766)
    zn = NODE_REDSHIFTS
    i = np.searchsorted(zn, z)
    d = cosmo.comoving_distance
    table = (d(zn[i]) - d(z)) / (d(zn[i]) - d(zn[i - 1]))
    assert np.abs(node_weight(z, 0.27) - table).max() < 1e-5
    # w_chi = w_z + O(node step): below 0.004 on the suite's grid, and ten times finer nodes give a tenth of it.
    coarse = np.abs(node_weight(z, 0.27) - node_weight(z, 0.27, "z")).max()
    fine_nodes = 5.0 * 1.002 ** np.arange(1001) - 1.0
    fine = np.abs(node_weight(z, 0.27, nodes=fine_nodes) - node_weight(z, 0.27, "z", nodes=fine_nodes)).max()
    assert 1e-3 < coarse < 4e-3 and 0.08 < fine / coarse < 0.12


def test_chi_weight_on_the_distance_grid_matches_chi_of_exact_redshifts():
    cosmo = FlatLCDM(0.33, 0.6766)
    los = np.arange(0, 2000, 7) * LC_CELL_MPC
    z = cosmo.z_at_distance(cosmo.comoving_distance(LC_Z_MIN) + los)
    assert np.abs(node_weight(z, 0.33, los_distance=los) - node_weight(z, 0.33)).max() < 1e-5
    # A redshift error of 1e-3 moves chi(z) but not the distance-grid weight.
    assert np.abs(node_weight(z + 1e-3, 0.33, los_distance=los) - node_weight(z, 0.33, los_distance=los)).max() == 0


def test_node_weight_per_slice_omega_m():
    z = np.array([7.01, 7.05, 9.3])
    om = np.array([0.25, 0.35, 0.25])
    per = node_weight(z, om)
    assert np.allclose(per, [node_weight(zi, oi) for zi, oi in zip(z, om)], rtol=0, atol=1e-15)


def _logits(shape, scale=3.0, seed=0):
    return scale * torch.randn(shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


def test_two_node_head_with_equal_logits_is_the_plain_head():
    out = _logits((5, 3, 4, 4))
    w = torch.tensor([0.0, 0.2, 0.5, 0.9, 1 - 1e-9], dtype=torch.float64)
    equal = out.clone()
    equal[:, 2] = out[:, 0]  # l_b = l_a with the free (non-monotone) head
    assert torch.allclose(two_node_output(equal, w, monotone=False), out[:, :2], rtol=0, atol=1e-12)
    # w = 0 is node a alone, whatever l_b.
    assert torch.allclose(two_node_output(out, torch.zeros(5, dtype=torch.float64))[:, 0], out[:, 0], atol=1e-12)
    assert torch.equal(two_node_output(out, w)[:, 1], out[:, 1])


def test_two_node_head_mixes_the_node_probabilities():
    out = _logits((6, 3, 4, 4), seed=1)
    w = torch.rand(6, dtype=torch.float64)
    for monotone in (True, False):
        l_b = out[:, 0] - F.softplus(out[:, 2]) if monotone else out[:, 2]
        x = (1 - w.view(-1, 1, 1)) * out[:, 0].sigmoid() + w.view(-1, 1, 1) * l_b.sigmoid()
        assert torch.allclose(two_node_output(out, w, monotone)[:, 0].sigmoid(), x, rtol=0, atol=1e-12)
    x = two_node_output(out, w)[:, 0].sigmoid()  # monotone: never above node a's probability
    assert torch.all(x <= out[:, 0].sigmoid() + 1e-15)


def test_log_space_bce_matches_naive_bce_and_stays_finite_at_saturation():
    out = _logits((6, 3, 4, 4), seed=2)
    w = torch.rand(6, dtype=torch.float64)
    y = torch.rand(6, 1, 4, 4, dtype=torch.float64)
    logit = two_node_output(out, w)[:, :1]
    x = logit.sigmoid()
    naive = -(y * x.log() + (1 - y) * (1 - x).log()).mean()
    assert torch.allclose(F.binary_cross_entropy_with_logits(logit, y), naive, rtol=1e-12)
    # Saturated float32 logits: the naive BCE overflows, the log-space one and its gradients stay finite.
    sat = torch.tensor([60.0, -60.0, 30.0, -30.0], dtype=torch.float32).view(4, 1, 1, 1).repeat(1, 3, 2, 2)
    sat.requires_grad_(True)
    for wv in (0.0, 0.3, 1.0):
        w = torch.full((4,), wv)
        logit = two_node_output(sat, w)[:, :1]
        for target in (0.0, 1.0):
            y = torch.full_like(logit, target)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            grad, = torch.autograd.grad(loss, sat, retain_graph=True)
            assert torch.isfinite(loss) and torch.isfinite(grad).all()
    naive = -torch.log(1 - sat[:, :1].detach().sigmoid())  # w = 0: x = sigmoid(l_a) rounds to 1 at l_a = 60
    assert not torch.isfinite(naive).all()


def test_node_options_off_leave_scalars_and_output_untouched():
    out, scal = torch.randn(2, 2, 4, 4), torch.randn(2, 5)
    assert node_scalars(scal, None, None) is scal and node_output(out, None, None) is out
    node = {"w_mode": "chi", "input": True, "two_node": False, "monotone": True}
    w = torch.tensor([0.25, 0.75])
    assert torch.equal(node_scalars(scal, w, node), torch.cat([scal, w[:, None]], 1))
    assert node_output(out, w, node) is out


def write_mf_cache(path, n_cones=6, per_cone=4, hw=8, n_cond=3, seed=0):
    rng = np.random.default_rng(seed)
    n = n_cones * per_cone
    stats = {"omm_mean": 0.3, "omm_std": 0.05}
    names = ["1/(1+z)", "OMm", "x_HI_global(0, 1)"]
    omm = np.repeat(rng.uniform(0.22, 0.38, n_cones), per_cone)
    los = rng.integers(100, 800, n)
    z = np.concatenate([FlatLCDM(om, 0.6766).z_at_distance(FlatLCDM(om, 0.6766).comoving_distance(LC_Z_MIN)
                                                            + los[k:k + per_cone] * LC_CELL_MPC)
                        for k, om in zip(range(0, n, per_cone), omm[::per_cone])])
    xhi = rng.uniform(0, 1, (n, hw, hw))
    with h5py.File(path, "w") as h:
        h["cond"] = rng.normal(size=(n, n_cond, hw, hw)).astype(np.float16)
        h["phys"] = rng.uniform(5, 30, (n, hw, hw)).astype(np.float32)
        h["xhi"] = xhi.astype(np.float16)
        h["tb"] = rng.normal(size=(n, hw, hw)).astype(np.float16)
        h["scalars"] = np.stack([1 / (1 + z), (omm - 0.3) / 0.05, xhi.mean((1, 2))], 1).astype(np.float32)
        h["z"] = z
        h["los_index"] = los.astype(np.int32)
        h["cone_id"] = np.repeat(np.arange(n_cones), per_cone)
        h["split"] = np.repeat(np.array([0, 0, 0, 0, 1, 2][:n_cones], np.int8), per_cone)
        h.attrs.update({"schema": SCHEMA, "scalar_names": json.dumps(names),
                        "structured": json.dumps({"stats": stats}),
                        "normalization": json.dumps({"brightness_temp": {"offset": -20.0, "scale": 50.0}})})
    return omm, z, los


def test_memory_split_node_phase_is_computed_from_the_cache(tmp_path):
    omm, z, los = write_mf_cache(tmp_path / "mf.h5")
    split = MFMemorySplit(tmp_path / "mf.h5", "train", w_mode="chi")
    expect = node_weight(z[:16], omm[:16], los_distance=los[:16] * LC_CELL_MPC)
    assert np.allclose(split.w.numpy(), expect, rtol=0, atol=1e-6)
    assert np.allclose(split.w.numpy(), node_weight(z[:16], omm[:16]), rtol=0, atol=1e-5)
    assert np.allclose(MFMemorySplit(tmp_path / "mf.h5", "train", w_mode="z").w.numpy(),
                       node_weight(z[:16], mode="z"), rtol=0, atol=1e-6)
    assert torch.equal(split.batch([3, 1])["w"], split.w[[3, 1]])
    assert "w" not in MFMemorySplit(tmp_path / "mf.h5", "train").batch([0])


TINY = dict(max_steps=4, min_steps=2, val_every=2, batch_size=4, width=8, modes=2, updates=2, threads=1,
            storage="cpu", lr=1e-3, min_lr=1e-5)


@pytest.mark.parametrize("flags", [dict(node_phase_input=True), dict(two_node_head=True),
                                   dict(node_phase_input=True, two_node_head=True, node_monotone=False, w_mode="z")])
def test_node_phase_training_smoke(tmp_path, flags):
    write_mf_cache(tmp_path / "mf.h5")
    run(tmp_path / "mf.h5", tmp_path / "run", MFConfig(**TINY, **flags), device="cpu")
    best = torch.load(tmp_path / "run" / "best.pt", weights_only=False)
    node = best["node_phase"]
    assert node["input"] == flags.get("node_phase_input", False) and node["w_mode"] == flags.get("w_mode", "chi")
    assert best["model_config"]["out_channels"] == (3 if flags.get("two_node_head") else 2)
    assert best["model_config"]["scalar_dim"] == 3 + node["input"]
    model = build_model(best["model_config"])
    model.load_state_dict(best["model"])
    split = MFMemorySplit(tmp_path / "mf.h5", "val", w_mode=node["w_mode"])
    scores = evaluate(model, split.loader(4), "cpu", best["head"], best["tb_norm"], node=node)
    assert all(np.isfinite(v) for v in scores.values())


def test_flags_off_checkpoint_has_no_node_phase_and_reloads(tmp_path):
    write_mf_cache(tmp_path / "mf.h5")
    run(tmp_path / "mf.h5", tmp_path / "run", MFConfig(**TINY), device="cpu")
    best = torch.load(tmp_path / "run" / "best.pt", weights_only=False)
    assert "node_phase" not in best and "node_phase" not in json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert best["model_config"]["out_channels"] == 2 and best["model_config"]["scalar_dim"] == 3
    model = build_model(best["model_config"])
    model.load_state_dict(best["model"])
    split = MFMemorySplit(tmp_path / "mf.h5", "val")
    batch = split.batch(np.arange(len(split)))
    with torch.no_grad():
        plain = model(batch["cond"], batch["scalars"])
        loader = split.loader(len(split))
        scores = evaluate(model, loader, "cpu", best["head"], best["tb_norm"])
    bce = float(F.binary_cross_entropy_with_logits(plain[:, :1], batch["target"][:, :1]))
    assert scores["bce"] == pytest.approx(bce, rel=1e-6)
