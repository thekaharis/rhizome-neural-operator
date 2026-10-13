"""Cylindrical (k_perp, k_par) power error maps (x_HI or T_b) on a linear scale, P_pred/P_true - 1,
with a flat white dead band.

    python -m ebm21cm.plot_cyl_error --out runs/eval3d/cyl_error_linear \\
        --model runs/eval3d/slicewise2d_mf_all_vs_mf:mf_cnn_whno_es_tbs_histtb_tbclean_ep26=UFNO \\
        --model runs/eval3d/slicewise2d_mf_xhi_power_loss:rhizome2d_mf_w64_all=Rhizome ...

Reads the cone-median cylindrical power ratio (``cyl_ratio_med``) that fno-21cm's
``power_spectrum_evaluation`` saved in each suite's ``power_spectrum/ps_results.npz``;
nothing is recomputed. The models must come from suites on the same cones and grids
(checked: wavenumber edges and redshift chunks must match). A results ``.npz`` in the
same layout (e.g. from ``ebm21cm.tb_cyl_power``) can stand in for a suite directory.

Colour: |error| <= ``--dead`` is flat white; beyond it the diverging map starts at a
visibly tinted level and saturates at +-``--vmax`` (red: too much power). Grey: no
truth power. Log wavenumber axes; bins with no Fourier modes in any model (e.g. a log k_par
bin between two LOS harmonics) are merged into the next larger bin, so every cell is a
logarithmic bin with data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def dead_band_cmap(vmax, dead, base="RdBu_r", start=0.12, n=1024):
    """Diverging colormap on [-vmax, vmax]: white for |v| <= dead, then base from 0.5 +- start outward."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = plt.get_cmap(base)
    v = np.linspace(-vmax, vmax, n)
    frac = np.clip((np.abs(v) - dead) / (vmax - dead), 0, 1)
    pos = 0.5 + np.sign(v) * (start + (0.5 - start) * frac)
    colors = cmap(pos)
    colors[np.abs(v) <= dead] = (1.0, 1.0, 1.0, 1.0)
    out = ListedColormap(colors)
    out.set_bad("0.85")
    return out


def merge_empty_bins(edges, maps, axis):
    """Absorb bins that are empty in every map (no Fourier modes) into the next larger bin.

    ``maps``: arrays sharing ``edges`` along ``axis`` (0: k_par rows, 1: k_perp columns). A bin
    with no modes carries no data, so the merged bin keeps exactly the next bin's values; the
    last bin is absorbed into the previous one instead. Returns (edges, maps).
    """
    edges = np.asarray(edges, float)
    maps = [np.asarray(m, float) for m in maps]
    empty = np.all([np.all(np.isnan(m), axis=1 - axis) for m in maps], axis=0)
    keep_bins = ~empty
    if not keep_bins.any() or not empty.any():
        return edges, maps
    keep_edges = np.ones(len(edges), bool)
    for i in np.flatnonzero(empty):
        keep_edges[i + 1 if i + 1 < len(edges) - 1 else i] = False
    return edges[keep_edges], [np.compress(keep_bins, m, axis=axis) for m in maps]


def load(spec):
    suite, _, rest = spec.partition(":")
    name, _, label = rest.partition("=")
    path = Path(suite) if suite.endswith(".npz") else Path(suite) / "power_spectrum" / "ps_results.npz"
    with np.load(path, allow_pickle=True) as d:
        if f"{name}/cyl_ratio_med" not in d.files:
            raise SystemExit(f"{suite}: no cylindrical results for {name}")
        return label or name, {k: d[f"{name}/{k}"] for k in ("cyl_ratio_med", "cyl_z_centers", "k_edges",
                                                                "cyl_kpar_edges")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", action="append", required=True, help="SUITE_DIR (or results .npz):model_name=Label")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vmax", type=float, default=0.4)
    ap.add_argument("--dead", type=float, default=0.1)
    ap.add_argument("--field", default=r"$x_{\rm HI}$", help="field name in the title")
    ap.add_argument("--prefix", default="ps_cyl_error", help="output file prefix")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [load(s) for s in a.model]
    ref = models[0][1]
    for label, res in models[1:]:
        for k in ("cyl_z_centers", "k_edges", "cyl_kpar_edges"):
            if not np.allclose(res[k], ref[k]):
                raise SystemExit(f"{label}: {k} differs from {models[0][0]}")
    cmap = dead_band_cmap(a.vmax, a.dead)
    a.out.mkdir(parents=True, exist_ok=True)
    zc, kperp = ref["cyl_z_centers"], ref["k_edges"]
    summary = {}
    fig, axes = plt.subplots(len(models), len(zc), figsize=(4.4 * len(zc), 3.5 * len(models)), squeeze=False,
                             constrained_layout=True)
    for col, z in enumerate(zc):
        errs = [res["cyl_ratio_med"][col] - 1.0 for _, res in models]
        kpar, errs = merge_empty_bins(ref["cyl_kpar_edges"][col], errs, axis=0)
        kp, errs = merge_empty_bins(kperp, errs, axis=1)
        for row, ((label, res), err) in enumerate(zip(models, errs)):
            ax = axes[row][col]
            err = np.ma.masked_invalid(err)
            pcm = ax.pcolormesh(kp, kpar, err, cmap=cmap, vmin=-a.vmax, vmax=a.vmax)
            ax.set_xscale("log")
            ax.set_yscale("log")
            inside = float(np.mean(np.abs(err.compressed()) <= a.dead))
            summary.setdefault(label, [None] * len(zc))[col] = inside
            ax.set_title(f"z ≈ {z:.0f}: {100 * inside:.0f}% of cells within ±{a.dead:g}", fontsize=10)
            if row == len(models) - 1:
                ax.set_xlabel(r"$k_\perp$  [Mpc$^{-1}$]")
            if col == 0:
                ax.set_ylabel(f"{label}\n" + r"$k_\parallel$  [Mpc$^{-1}$]", fontsize=10)
    bar = fig.colorbar(pcm, ax=axes, shrink=0.6, aspect=30, extend="both")
    bar.set_label(r"$P_{\rm pred}/P_{\rm true} - 1$  (white: within $\pm$" + f"{a.dead:g})")
    ticks = sorted({-a.vmax, -a.dead, 0.0, a.dead, a.vmax} | set(np.round(np.arange(-a.vmax, a.vmax + 1e-9, 0.2), 2)))
    bar.set_ticks(ticks)
    fig.suptitle(f"Cylindrical {a.field} power error, cone median over 199 test cones")
    path = a.out / f"{a.prefix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)
    for label, fr in summary.items():
        print(f"{label:28s} share of (k_perp, k_par) cells within ±{a.dead:g}: "
              + " ".join(f"z≈{z:.0f} {100 * f:.0f}%" for z, f in zip(zc, fr)))


if __name__ == "__main__":
    main()
