"""Metrics for an ensemble of generated slices against one truth.

Point metrics (RMSE of a single sample and of the ensemble mean) are kept
for comparison with deterministic regressors, but a generative model is
judged by proper scores and by statistics of individual samples:

* CRPS (fair ensemble estimator) and the spread/skill ratio (~1 when
  calibrated).
* Transverse power spectrum ratio P_sample / P_truth and cross-correlation
  r(k) -- a conditional-mean predictor has ratio < 1 at bubble scales.
* x_HI "hedging fraction": pixels with 0.1 < x_HI < 0.9. Truth is almost
  binary; an L2 regressor fills fronts with intermediate values.
* Ionized-region size distribution via the mean-free-path method on the
  binarized field (x_HI < 0.5), compared by the 1-D Wasserstein distance of
  log sizes.

All inputs are numpy arrays in physical units; ``cell`` is in Mpc.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import wasserstein_distance


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - b) ** 2)))


def crps_ensemble(samples, truth):
    """Fair CRPS averaged over pixels. samples: (M, ...), truth: (...)."""
    s = np.asarray(samples, np.float64)
    m = s.shape[0]
    t1 = np.abs(s - truth[None]).mean(0)
    if m < 2:
        return float(t1.mean())
    ss = np.sort(s, axis=0)
    # E|X - X'| over distinct pairs, via the sorted-sample identity.
    w = (2 * np.arange(1, m + 1) - m - 1).reshape((m,) + (1,) * (s.ndim - 1))
    pair = 2.0 * (w * ss).sum(0) / (m * (m - 1))
    return float((t1 - 0.5 * pair).mean())


def spread_skill(samples, truth):
    """Ensemble std over RMSE of the mean, with the finite-M correction."""
    s = np.asarray(samples, np.float64)
    m = s.shape[0]
    if m < 2:
        return float("nan")
    spread = np.sqrt(s.var(0, ddof=1).mean())
    skill = np.sqrt(((s.mean(0) - truth) ** 2).mean())
    return float(spread / skill * np.sqrt((m + 1) / m)) if skill > 0 else float("nan")


def hedging_fraction(xhi, lo=0.1, hi=0.9):
    x = np.asarray(xhi)
    return float(((x > lo) & (x < hi)).mean())


def _kgrid(shape, cell):
    ky = 2 * np.pi * np.fft.fftfreq(shape[0], d=cell)
    kx = 2 * np.pi * np.fft.rfftfreq(shape[1], d=cell)
    k = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    # rfft double-counts all columns except kx=0 and (even) Nyquist.
    w = np.full(k.shape, 2.0)
    w[:, 0] = 1.0
    if shape[1] % 2 == 0:
        w[:, -1] = 1.0
    return k, w


def k_bins(shape, cell, n_bins=None):
    k_nyq = np.pi / cell
    k_f = 2 * np.pi / (shape[0] * cell)
    n = n_bins or max(4, shape[0] // 4)
    return np.linspace(k_f * 0.5, k_nyq, n + 1)


def cross_power(a, b, cell, edges):
    """Binned (P_ab, P_aa, P_bb) of mean-removed 2-D fields, batched over leading dims."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    b = b - b.mean(axis=(-2, -1), keepdims=True)
    fa, fb = np.fft.rfft2(a), np.fft.rfft2(b)
    k, w = _kgrid(a.shape[-2:], cell)
    idx = np.digitize(k.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    area = (a.shape[-2] * cell) * (a.shape[-1] * cell)
    norm = cell ** 4 / area

    # Weighted one-hot bin matrix: binned = (field * w) summed per bin / sum(w).
    onehot = np.zeros((int(ok.sum()), len(edges) - 1))
    onehot[np.arange(onehot.shape[0]), idx[ok]] = w.ravel()[ok]
    cnt = onehot.sum(0)

    def binned(x):
        x = x.reshape(*x.shape[:-2], -1)[..., ok]
        with np.errstate(invalid="ignore", divide="ignore"):
            return (x @ onehot) / cnt * norm

    return binned((fa * fb.conj()).real), binned(np.abs(fa) ** 2), binned(np.abs(fb) ** 2)


def spectra(samples, truth, cell, edges):
    """Mean P_sample/P_truth and mean r(k) between each sample and the truth."""
    s = np.asarray(samples, np.float64)
    pab, paa, pbb = cross_power(s, np.broadcast_to(truth, s.shape), cell, edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = paa / pbb
        r = pab / np.sqrt(paa * pbb)
    pm_ab, pm_aa, pm_bb = cross_power(s.mean(0), truth, cell, edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_mean = pm_aa / pm_bb
        r_mean = pm_ab / np.sqrt(pm_aa * pm_bb)
    with warnings.catch_warnings():  # constant fields (fully neutral/ionized) give all-NaN bins
        warnings.simplefilter("ignore", RuntimeWarning)
        ratio, r = np.nanmean(ratio, 0), np.nanmean(r, 0)
    return {"ratio": ratio, "r": r,
            "ratio_ens_mean": ratio_mean, "r_ens_mean": r_mean, "p_truth": pbb[0]}


def mfp_sizes(ionized, cell, n_rays=2000, rng=None, max_cells=None):
    """Mean-free-path region sizes (Mpc) of a boolean 2-D periodic map."""
    ion = np.asarray(ionized, bool)
    H, W = ion.shape
    starts = np.argwhere(ion)
    if starts.size == 0:
        return np.zeros(0)
    rng = rng or np.random.default_rng(0)
    pick = starts[rng.integers(len(starts), size=n_rays)].astype(np.float64)
    ang = rng.uniform(0, 2 * np.pi, n_rays)
    step = np.stack([np.sin(ang), np.cos(ang)], 1)
    max_cells = max_cells or max(H, W)
    length = np.full(n_rays, float(max_cells))
    alive = np.ones(n_rays, bool)
    pos = pick + 0.5
    for n in range(1, max_cells + 1):
        pos = pos + step
        iy = np.floor(pos[:, 0]).astype(int) % H
        ix = np.floor(pos[:, 1]).astype(int) % W
        hit = alive & ~ion[iy, ix]
        length[hit] = n
        alive &= ~hit
        if not alive.any():
            break
    return length * cell


def size_distance(sizes_a, sizes_b):
    if len(sizes_a) == 0 or len(sizes_b) == 0:
        return float("nan")
    return float(wasserstein_distance(np.log(sizes_a), np.log(sizes_b)))
