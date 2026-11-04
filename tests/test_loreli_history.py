import h5py
import numpy as np
import pytest
import torch

from ebm21cm.data.loreli_globals import bin_mean, grid_params, plane_run
from ebm21cm.data.loreli_memory import SIDE_SCHEMA, side_scalars
from ebm21cm.loreli_history import Net, features, midpoint

SPEC = {
    "RH": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "FX": [0.1, 0.15, 0.22, 0.32, 0.46, 0.68, 1.0, 1.47, 2.15, 3.16, 4.64, 6.81, 10.0],
    "FE": [0.275, 0.05, 0.5],
    "PAIRS": [[8191.9 + k, 8.0 + 0.01 * k] for k in range(42)],
}


def test_grid_rule_strides():
    # r_H/S fastest (6), then f_x (13, stride 6), then (tau, Mmin) (42, stride 78), f_esc slowest (3276)
    assert grid_params(10001, SPEC) == (0.1, 8191.9, 0.2, 8.0, 0.275)
    assert grid_params(10006, SPEC)[0] == 0.15 and grid_params(10006, SPEC)[2] == 0.0
    assert grid_params(10078, SPEC)[1] == 8192.9
    assert grid_params(10000 + 3276, SPEC)[4] == 0.05
    block = [grid_params(10000 + 6 * 7 + k, SPEC) for k in range(6)]
    assert len({(p[0], p[1], p[3], p[4]) for p in block}) == 1      # a split block varies only r_H/S


def test_plane_run_trims_edges_and_keeps_longest_run():
    ok = np.zeros(20, bool)
    ok[2:5] = True
    ok[8:18] = True
    keep = plane_run(ok)
    assert np.flatnonzero(keep).tolist() == list(range(9, 17))


def test_bin_mean_partial_bins():
    v = np.arange(16, dtype=float)
    valid = np.ones(16, bool)
    valid[:5] = False                       # 3 of 8 valid: below half a bin
    out = bin_mean(v, valid, size=8)
    assert np.isnan(out[0]) and out[1] == pytest.approx(11.5)
    valid[:5] = True
    valid[0] = False                        # an end bin with its trimmed edge plane still counts
    assert bin_mean(v, valid, size=8)[0] == pytest.approx(4.0)


def test_monotone_head_rises_along_nodes():
    torch.manual_seed(0)
    net = Net(5, 32, monotone=True)
    y = net(torch.randn(7, 5))
    assert y.shape == (7, 32) and bool((y[:, 1:] >= y[:, :-1]).all())


def test_features_and_midpoint():
    f = features(np.array([[10.0, 1000.0, 0.4, 8.5, 0.05]]))
    assert np.allclose(f, [[1.0, 3.0, 0.4, 8.5, 0.05]])
    z = np.linspace(5, 15, 11)
    x = np.clip((z - 6.5) / 4, 0, 1)[None]
    assert midpoint(x, z)[0] == pytest.approx(8.5)


def test_side_scalars_checks_rows(tmp_path):
    path = tmp_path / "side.h5"
    with h5py.File(path, "w") as h:
        h["scalars"] = np.ones((3, 38), np.float32)
        h["cone_id"] = np.array([1, 1, 2])
        h["los_index"] = np.array([0, 1, 0])
        h.attrs["schema"] = SIDE_SCHEMA
    assert side_scalars(path, np.array([1, 1, 2]), np.array([0, 1, 0])).shape == (3, 38)
    with pytest.raises(ValueError):
        side_scalars(path, np.array([1, 2, 2]), np.array([0, 1, 0]))
