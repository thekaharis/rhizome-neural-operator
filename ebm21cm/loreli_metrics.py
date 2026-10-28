"""Metrics for a slice-wise prediction on a non-periodic LoReLi plane.

All fields are physical (x_HI in [0, 1], T_b in mK) and evaluated on the native
128-cell LoReLi grid (2.297 Mpc). The shape metrics borrow the project's
existing mean-free-path bubble sizes and ionized-fraction-matched IoU; the
front profile uses the signed distance to the truth's x_HI = 0.5 contour, as
in ``ebm21cm.edge_compare``. The plane is not periodic, so every metric is
computed on a centred crop with a fixed rectangular mask.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.stats import wasserstein_distance

from . import metrics

STAGES = (("early", 0.05, 0.35), ("mid", 0.35, 0.65), ("late", 0.65, 0.95))


def stage_of(mean_xhi):
    """Stage name for a slice mean x_HI, or None outside the active range."""
    for name, lo, hi in STAGES:
        if lo <= mean_xhi < hi:
            return name
    return None


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred, np.float64) - true) ** 2)))


def smooth(field, sigma_mpc, cell):
    """Gaussian smoothing with sigma in Mpc; no smoothing at sigma = 0."""
    return field if sigma_mpc <= 0 else gaussian_filter(field, sigma_mpc / cell, mode="nearest")


def iou_matched(pred, true, threshold=0.5):
    """IoU of the ionized masks with the prediction thresholded at the truth's ionized fraction."""
    ion = true < threshold
    fraction = float(ion.mean())
    if not 0 < fraction < 1:
        return float("nan")
    cut = np.quantile(pred, 1.0 - fraction)
    guess = pred <= cut
    union = np.count_nonzero(ion | guess)
    return float(np.count_nonzero(ion & guess) / union) if union else float("nan")


def power_edges(kmin=0.05, kmax=0.6, n=10):
    return np.linspace(kmin, kmax, n + 1)


def binned_power(pred, true, cell, edges):
    """(P_ab, P_aa, P_bb) for one plane, from :func:`ebm21cm.metrics.cross_power`."""
    return metrics.cross_power(pred, true, cell, edges)


def bubble_sizes(xhi, cell, threshold=0.5, n_rays=1000, seed=0):
    """Mean-free-path ionized-region sizes (Mpc) for x_HI below ``threshold``."""
    return metrics.mfp_sizes(xhi < threshold, cell, n_rays=n_rays, rng=np.random.default_rng(seed))


def front_distance(true, cell, threshold=0.5):
    """Signed distance (Mpc) to the nearest opposite-phase cell; positive in neutral gas."""
    neutral = true > threshold
    if neutral.all() or not neutral.any():
        return None
    return np.where(neutral,
                    distance_transform_edt(~neutral, sampling=cell),
                    -distance_transform_edt(neutral, sampling=cell))


def front_profile(pred, true, cell, edges):
    """(counts, sum_true, sum_pred) in signed-distance bins from the truth contour; None if no front."""
    distance = front_distance(true, cell)
    if distance is None:
        return None
    index = np.digitize(distance.ravel(), edges) - 1
    ok = (index >= 0) & (index < len(edges) - 1)
    n = len(edges) - 1
    return (np.bincount(index[ok], minlength=n),
            np.bincount(index[ok], weights=np.asarray(true).ravel()[ok], minlength=n),
            np.bincount(index[ok], weights=np.asarray(pred).ravel()[ok], minlength=n))


def front_width(distance, profile, lo=0.1, hi=0.9):
    """10-90 transition width (Mpc) of a mean x_HI profile across a front."""
    m = np.isfinite(profile)
    d, f = np.asarray(distance)[m], np.asarray(profile)[m]
    order = np.argsort(d)
    d, f = d[order], f[order]
    if len(f) < 2 or f.min() > lo or f.max() < hi:
        return float("nan")

    def crossing(level):
        idx = np.where(np.diff(np.sign(f - level)) != 0)[0]
        if len(idx) == 0:
            return float("nan")
        i = idx[0]
        return d[i] + (level - f[i]) * (d[i + 1] - d[i]) / (f[i + 1] - f[i]) if f[i + 1] != f[i] else d[i]

    return float(crossing(hi) - crossing(lo))


def tb_pdf_w1(pred, true, max_cells=20000):
    p, t = np.asarray(pred).ravel(), np.asarray(true).ravel()
    step = max(1, p.size // max_cells)
    return float(wasserstein_distance(p[::step], t[::step]))


def sharp_k_max(delta, cell, radii=None, threshold_radius=None):
    """max_R delta_R with a 2-D sharp-k filter |k| R <= 1, on a mirror-padded non-periodic plane."""
    from .data.loreli import mirror_pad

    radii = np.geomspace(0.9, 31.0, 16) if radii is None else np.asarray(radii, float)
    pad = int(np.ceil(radii.max() / cell)) + 2
    field = mirror_pad(mirror_pad(np.asarray(delta, np.float32), pad, 0), pad, 1)
    spectrum = np.fft.rfft2(field)
    ny, nx = field.shape
    kx = 2 * np.pi * np.fft.fftfreq(ny, d=cell)
    ky = 2 * np.pi * np.fft.rfftfreq(nx, d=cell)
    k = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    best = np.full(field.shape, -np.inf, np.float32)
    for radius in radii:
        smooth = np.fft.irfft2(spectrum * (k * radius <= 1.0), s=field.shape)
        best = np.maximum(best, smooth)
    return best[pad:-pad, pad:-pad]
