import numpy as np
import pytest

from ebm21cm import metrics as M


def test_crps():
    rng = np.random.default_rng(0)
    t = rng.standard_normal((8, 8))
    assert M.crps_ensemble(np.stack([t] * 5), t) == pytest.approx(0.0, abs=1e-12)
    # Matches the brute-force pairwise definition.
    s = rng.standard_normal((6, 8, 8))
    brute = (np.abs(s - t).mean(0) - 0.5 * np.abs(s[:, None] - s[None]).sum((0, 1)) / (6 * 5)).mean()
    assert M.crps_ensemble(s, t) == pytest.approx(brute)


def test_spread_skill_calibrated():
    rng = np.random.default_rng(1)
    center = rng.standard_normal((64, 64))
    truth = center + rng.standard_normal((64, 64))
    ens = center + rng.standard_normal((32, 64, 64))
    assert M.spread_skill(ens, truth) == pytest.approx(1.0, abs=0.05)


def test_white_noise_spectrum_flat_and_r():
    rng = np.random.default_rng(2)
    cell = 2.0
    a = rng.standard_normal((64, 128, 128))
    edges = M.k_bins((128, 128), cell)
    pab, paa, pbb = M.cross_power(a, a, cell, edges)
    p = np.nanmean(paa, 0)
    assert np.nanstd(p[1:]) / np.nanmean(p[1:]) < 0.05
    assert np.nanmean(p) == pytest.approx(cell ** 2, rel=0.05)  # white noise, unit variance
    assert np.allclose(pab / np.sqrt(paa * pbb), 1.0)


def test_hedging():
    assert M.hedging_fraction(np.array([0.0, 1.0, 0.5, 0.05])) == 0.25


def test_mfp_disk():
    H, r = 128, 20
    yy, xx = np.mgrid[:H, :H]
    disk = (yy - 64) ** 2 + (xx - 64) ** 2 < r ** 2
    s = M.mfp_sizes(disk, 1.0, n_rays=4000, rng=np.random.default_rng(0))
    # Rays from uniform interior points of a disk: mean chord-to-edge is 2r*... ~ between r/2 and r.
    assert 0.5 * r < s.mean() < 1.0 * r
    assert M.size_distance(s, s) == 0.0
