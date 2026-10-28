import json

import h5py
import numpy as np
import pytest
import torch

from ebm21cm import loreli_metrics as lm
from ebm21cm.data.loreli import (LoreliAdapter, LoreliCone, band_planes, linear_velocity, loreli_redshifts,
                                 mirror_pad, pad_to, resample_transverse, training_stats)
from ebm21cm.loreli_inference import area_matrix, downsample_transverse, load_model, map_to_native
from ebm21cm.model.remap import remapped_model, remap_spectral_conv
from ebm21cm.model.rhizome import build_model

STATS = {"bands": [(-4, -2), (-1, 0), (0, 1), (2, 4)],
         "normalization": {"density": {"offset": 0.0, "scale": 10.0},
                           "los_velocity": {"offset": 0.0, "scale": 1e-17}},
         "param_names": [f"p{i}" for i in range(11)],
         "scalar_names": ["1/(1+z)"] + [f"p{i}" for i in range(11)] + [f"h{i}" for i in range(8)]}


def write_cone(path, n_native=16, n_los=32, seed=0, tail=4):
    """A tiny five-field file: index 0 is low z, a NaN/zero tail at the low-z end."""
    rng = np.random.default_rng(seed)
    cube = np.zeros((n_native, n_native, n_los, 5), np.float32)
    cube[..., 0] = 1.0
    cube[..., 1] = 100.0
    cube[..., 2] = np.linspace(0.9, 0.0, n_los)[None, None, :]
    cube[..., 3] = (1 + 0.3 * rng.standard_normal((n_native, n_native, n_los))).astype(np.float32) * 1e-9
    cube[..., 4] = (-20 + 5 * rng.standard_normal((n_native, n_native, n_los))).astype(np.float32)
    if tail:
        cube[:, :, :tail, 3] = 0.0
        cube[:, :, :tail, 4] = np.nan
    cube.tofile(path)
    return cube


def test_redshifts_are_uniform_in_comoving_distance():
    z, distance = loreli_redshifts((5.0, 15.0), 1536)
    assert z[0] == pytest.approx(5.0, abs=1e-12) and z[-1] == pytest.approx(15.0, abs=1e-10)
    assert np.all(np.diff(z) > 0)
    from ebm21cm.cosmology import FlatLCDM

    step = np.diff(distance)
    exact = (FlatLCDM(0.308, 0.678).comoving_distance(15.0)
             - FlatLCDM(0.308, 0.678).comoving_distance(5.0)) / 1535
    assert step.min() == pytest.approx(step.max(), rel=1e-10)
    assert step.mean() == pytest.approx(exact, rel=1e-12)


def test_reader_orientation_and_validity_mask(tmp_path):
    path = tmp_path / "lightcone_xa_tpHI_xion_dens_dtb_no_rsd_simu00001.dat"
    write_cone(path, n_native=16, n_los=32, tail=4)
    cone = LoreliCone(path, z_range=(5.0, 5.4), n_native=16, n_los=32)
    # The empty low-z tail is dropped, index 0 stays the low-z end, and the run is contiguous.
    assert not cone.valid[:4].any()
    assert cone.valid[4:].all() and cone.n_slices == 28
    assert cone.z[0] < cone.z[-1]
    xhi, tb = cone.history()
    assert xhi[4] < xhi[-1]  # low z is more ionized
    fields = cone.read(4, 8)
    assert fields["xion"].shape == (16, 16, 4)
    assert np.allclose(1 - fields["xion"][..., 0], xhi[4])


def test_delta_has_zero_slice_mean(tmp_path):
    path = tmp_path / "lightcone_xa_tpHI_xion_dens_dtb_no_rsd_simu00002.dat"
    write_cone(path, n_native=12, n_los=20, tail=0)
    cone = LoreliCone(path, z_range=(5.0, 5.2), n_native=12, n_los=20, density_mean="slice")
    dens = cone.read(0, 20)["dens"]
    delta = cone.delta(dens)
    assert np.abs(delta.mean((0, 1))).max() < 1e-6
    assert delta.std() == pytest.approx(0.3, abs=0.03)


def test_mirror_pad_reflects_without_repeating_the_edge():
    x = np.arange(5)
    assert np.array_equal(mirror_pad(x, 2), [2, 1, 0, 1, 2, 3, 4, 3, 2])
    plane = np.arange(16).reshape(4, 4)
    padded = pad_to(plane, 8)
    assert padded.shape == (8, 8)
    assert np.array_equal(padded[2:6, 2:6], plane)
    assert padded[1, 2:6].tolist() == [4, 5, 6, 7]


def test_resample_transverse_keeps_a_constant_and_maps_a_ramp():
    rng = np.random.default_rng(0)
    constant = np.full((16, 16, 3), 7.0, np.float32)
    assert np.allclose(resample_transverse(constant, 21), 7.0)
    ramp = np.broadcast_to(np.arange(16, dtype=np.float32)[None, :, None], (16, 16, 2))
    out = resample_transverse(ramp, 32)
    expected = (np.arange(32) + 0.5) * 16 / 32 - 0.5
    assert np.abs(out[0, :, 0] - expected).max() < 0.6  # bicubic through a piecewise-linear ramp
    assert np.all(np.diff(out[0, :, 0]) > 0)


def test_band_planes_match_direct_means():
    rng = np.random.default_rng(1)
    values = rng.standard_normal((7, 5, 20)).astype(np.float32)
    bands = [(0, 1), (1, 3), (-1, 0)]
    centers = np.array([1, 5, 10])
    out = band_planes(values, centers, bands)
    for i, c in enumerate(centers):
        for b, (lo, hi) in enumerate(bands):
            assert np.allclose(out[i, b], values[:, :, c + lo:c + hi].mean(-1), atol=1e-6)


def test_linear_velocity_matches_the_analytic_mode():
    from ebm21cm.data.loreli import hubble_growth

    n, cell, z = 64, 2.0, 8.0
    k = 2 * np.pi * 3 / (n * cell)
    delta = np.cos(k * np.arange(n) * cell)[None, None, :].repeat(n, 0).repeat(n, 1).astype(np.float32)
    velocity = linear_velocity(delta, z, cell=cell)
    # For delta = cos(k z), v = i H f k delta_k / k^2 = -H f sin(k z) / k.
    expected = -hubble_growth(z) / k * np.sin(k * np.arange(n) * cell)
    assert np.allclose(velocity[0, 0], expected, rtol=2e-2, atol=1e-22)


def test_adapter_shapes_and_scalars(tmp_path):
    path = tmp_path / "lightcone_xa_tpHI_xion_dens_dtb_no_rsd_simu00003.dat"
    write_cone(path, n_native=16, n_los=32, tail=4)
    cone = LoreliCone(path, z_range=(5.0, 5.4), n_native=16, n_los=32)
    adapter = LoreliAdapter(cone, STATS, velocity="zero")
    inputs = adapter.inputs(2, 5)
    assert inputs["cond"].shape == (3, 2 * len(STATS["bands"]), 280, 280)
    assert inputs["scalars"].shape == (3, 1 + 11 + 2 * len(STATS["bands"]))
    assert inputs["phys"].shape == (3, 1, 280, 280)
    assert np.isfinite(inputs["cond"]).all() and np.isfinite(inputs["scalars"]).all()
    assert np.allclose(inputs["scalars"][:, 0], 1 / (1 + inputs["z"]), atol=1e-6)
    assert np.allclose(inputs["scalars"][:, 1:12], 0.0)
    history = inputs["scalars"][:, 12:12 + len(STATS["bands"])]
    assert np.all((history >= 0) & (history <= 1.01)) and history.max() > 0
    # No velocity: the last half of the conditioning is exactly zero.
    assert np.allclose(inputs["cond"][:, len(STATS["bands"]):], 0.0)


def test_area_matrix_rows_sum_to_one():
    weights = area_matrix(128, 206)
    assert weights.shape == (128, 206)
    assert np.allclose(weights.sum(1), 1.0, atol=1e-6)
    assert np.all(weights >= 0)


def test_map_to_native_preserves_a_constant_and_masks_the_border(tmp_path):
    path = tmp_path / "lightcone_xa_tpHI_xion_dens_dtb_no_rsd_simu00004.dat"
    write_cone(path, n_native=16, n_los=32, tail=0)
    cone = LoreliCone(path, z_range=(5.0, 5.4), n_native=16, n_los=32)
    adapter = LoreliAdapter(cone, STATS, velocity="zero")
    n = len(adapter.target_distance)
    xhi = np.full((n, 280, 280), 0.25, np.float32)
    tb = np.full_like(xhi, -3.0)
    out_x, out_t, keep = map_to_native(xhi, tb, adapter, cone, border=10)
    assert np.allclose(out_x[cone.valid], 0.25)
    assert np.allclose(out_t[cone.valid], -3.0)
    assert keep.sum() < 16 * 16


def test_downsample_transverse_keeps_the_axis_order(tmp_path):
    path = tmp_path / "lightcone_xa_tpHI_xion_dens_dtb_no_rsd_simu00005.dat"
    write_cone(path, n_native=16, n_los=32, tail=0)
    cone = LoreliCone(path, z_range=(5.0, 5.4), n_native=16, n_los=32)
    adapter = LoreliAdapter(cone, STATS, velocity="zero")
    ramp = (np.arange(280)[:, None] + 2 * np.arange(280)[None, :]).astype(np.float32)
    out = downsample_transverse(ramp, adapter, cone)
    weights = area_matrix(16, adapter.n_out)
    expected = weights @ ramp[37:243, 37:243] @ weights.T
    assert np.allclose(out, expected, atol=1e-4)


def _tiny_model_config():
    return {"in_channels": 4, "out_channels": 2, "scalar_dim": 3, "width": 8, "modes": 4,
            "n_steps": 3, "step_size": 0.5}


def test_remap_reproduces_a_2x2_tiled_prediction():
    torch.manual_seed(0)
    config = _tiny_model_config()
    native = build_model(config)
    model = remapped_model(config, native.state_dict(), factor=2)
    x = torch.randn(2, 4, 32, 32)
    scalars = torch.randn(2, 3)
    with torch.no_grad():
        a = native(x, scalars)
        tiled = torch.cat([torch.cat([x, x], -1)] * 2, -2)
        b = model(tiled, scalars)
    assert torch.allclose(a, b[:, :, :32, :32], atol=1e-6)
    assert torch.allclose(b[:, :, :32, :32], b[:, :, :32, 32:], atol=1e-6)
    assert torch.allclose(b[:, :, :32, :32], b[:, :, 32:, :32], atol=1e-6)


def test_remap_factor_one_is_the_identity():
    torch.manual_seed(1)
    config = _tiny_model_config()
    native = build_model(config)
    same = remapped_model(config, native.state_dict(), factor=1)
    x = torch.randn(2, 4, 32, 32)
    with torch.no_grad():
        assert torch.equal(native(x, torch.zeros(2, 3)), same(x, torch.zeros(2, 3)))


def test_remap_interpolates_odd_modes_and_copies_even_ones():
    rng = np.random.default_rng(0)
    modes = 3
    axis = torch.as_tensor(rng.standard_normal((2, 2, modes - 1)) + 0j)
    positive = torch.as_tensor(rng.standard_normal((2, 2, modes, modes - 1)) + 0j)
    negative = torch.as_tensor(rng.standard_normal((2, 2, modes - 1, modes - 1)) + 0j)
    dc = torch.as_tensor(rng.standard_normal((2, 2)))
    dc2, axis2, pos2, neg2 = remap_spectral_conv(dc, axis, positive, negative, modes, factor=2)
    assert axis2.shape == (2, 2, 2 * modes - 1)
    assert pos2.shape == (2, 2, 2 * modes, 2 * modes - 1)
    assert neg2.shape == (2, 2, 2 * modes - 1, 2 * modes - 1)
    assert torch.equal(dc2, dc)
    # index 0 is physical mode 1: even new index 1 maps to old index 0 exactly
    assert torch.allclose(axis2[:, :, 1], axis[:, :, 0])
    assert torch.allclose(axis2[:, :, 3], axis[:, :, 1])
    # odd new index 2 is halfway between old modes 1 and 2
    assert torch.allclose(axis2[:, :, 2], (axis[:, :, 0] + axis[:, :, 1]) / 2)
    assert torch.allclose(pos2[:, :, 0, 1], positive[:, :, 0, 0])
    assert torch.allclose(pos2[:, :, 0, 2], (positive[:, :, 0, 0] + positive[:, :, 0, 1]) / 2)
    assert torch.allclose(neg2[:, :, 1, 1], negative[:, :, 0, 0])


def test_load_model_refuses_node_phase(tmp_path):
    best = {"model_config": _tiny_model_config(), "model": build_model(_tiny_model_config()).state_dict(),
            "node_phase": {"input": True}}
    torch.save(best, tmp_path / "best.pt")
    with pytest.raises(SystemExit):
        load_model(tmp_path)


def test_stats_reader(tmp_path):
    path = tmp_path / "cache.h5"
    with h5py.File(path, "w") as h:
        h.attrs["bands"] = json.dumps([[-1, 0], [0, 1]])
        h.attrs["normalization"] = json.dumps({"density": {"offset": 0, "scale": 1}})
        h.attrs["param_names"] = json.dumps(["a", "b"])
        h.attrs["scalar_names"] = json.dumps(["1/(1+z)", "a", "b", "h", "h"])
    stats = training_stats(path)
    assert stats["bands"] == [(-1, 0), (0, 1)] and stats["param_names"] == ["a", "b"]


def test_metrics_on_a_synthetic_front():
    x = np.linspace(0, 1, 64)[:, None].repeat(64, 1)
    true = np.where(x < 0.5, 0.0, 1.0)
    pred = true.copy()
    assert lm.rmse(pred, true) == 0
    assert lm.iou_matched(pred, true) == 1
    noisy = np.clip(true + 0.2 * np.random.default_rng(0).standard_normal(true.shape), 0, 1)
    assert 0 < lm.iou_matched(noisy, true) < 1
    profile = lm.front_profile(true, true, 1.0, lm.FRONT_EDGES)
    assert profile is not None and profile[0].sum() > 0
