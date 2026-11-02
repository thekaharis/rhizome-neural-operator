"""Representative slices and cylindrical 2-D power spectra: reference vs a fine-tuned run, LoReLi v3 test.

    python docs/figures/loreli/src/make_loreli_slices_cyl.py [--models reference b8_pointwise]
        [--truth-root .../data/eval_cubes/loreli_l294_v3_test] [--preds runs/cm21eval/preds]
        [--groups runs/cm21eval/groups.json] [--out docs/figures/loreli/cm21eval]
        [--slice-cones 12316 19547] [--cones ...]

Inputs are the full test lightcones of the 21cmEVAL pipeline (truth from the per-group eval cubes,
predictions from ``scripts/loreli_cm21eval_predict.py``; interior 206 x 206 cells of 1.427 Mpc,
LOS on the 1.4286 Mpc comoving grid).  Writes

  slices_{xhi,tb}       truth / models / model - truth for two cones at x_HI-bar = 0.75, 0.5, 0.25
  lightcone_<cone>      transverse-LOS cuts through whole cones
  cyl_{xhi,tb}          cylindrical P(k_perp, k_par): truth Delta^2, P_model / P_truth and r per model,
                        in three stage-matched chunks
  cyl_summary.json      the binned maps

Cylindrical spectra follow ``ebm21cm.tb_cyl_power`` / ``native_cyl``: per-slice mean removed, Hann
window along the (non-periodic) LOS chunk, periodic transverse axes, ratio and r per cone, then the
median over cones.  Cones reionize at different z, so chunks are placed by stage rather than at a
fixed redshift: 128 planes (183 Mpc) centred where the cone's true slice-mean x_HI is 0.8 / 0.5 / 0.2
(chunks that do not fit the window or miss the stage by > 0.15 are skipped).
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

REPO = Path(__file__).resolve().parents[4]
W = Path("/pfs/10/work/hd_id260-fno_training")
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["reference", "b8_pointwise"])
ap.add_argument("--truth-root", default=str(W / "data/eval_cubes/loreli_l294_v3_test"))
ap.add_argument("--preds", default=str(REPO / "runs/cm21eval/preds"))
ap.add_argument("--groups", default=str(REPO / "runs/cm21eval/groups.json"))
ap.add_argument("--out", default=str(REPO / "docs/figures/loreli/cm21eval"))
ap.add_argument("--slice-cones", nargs="+", type=int, default=[12316, 19547])
ap.add_argument("--cones", nargs="+", type=int, default=None, help="subset for the spectra (default: all)")
ap.add_argument("--chunk", type=int, default=128)
a = ap.parse_args()
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
groups = json.loads(Path(a.groups).read_text())
group_of = {c: g for g, v in groups.items() for c in v["cones"]}
LABEL = {"reference": "reference (zero-shot)", "reference_oracle": "reference, oracle hist.",
         **{f"b{b}_{arm}": f"{b} blk: {arm}" for b in (8, 44) for arm in ("decoder", "ends", "pointwise", "none", "scratch")},
         "b44_none_oracle": "44 blk: none, oracle hist."}
lab = lambda m: LABEL.get(m, m)
L_PERP, CELL = 294.0, 1.4286
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "figure.dpi": 100})
warnings.simplefilter("ignore", RuntimeWarning)


def load(cone):
    g = group_of[cone]
    with np.load(Path(a.truth_root) / g / f"cone{cone}.npz") as d:
        truth = {"xhi": d["xhi"].astype(np.float32), "tb": d["tb"].astype(np.float32)}
        z = d["z_grid"].astype(float)
    preds = {}
    for m in a.models:
        with np.load(Path(a.preds) / m / f"cone{cone}.npz") as d:
            if not np.allclose(d["z_grid"], z):
                raise ValueError(f"{m}/cone{cone}: z grid differs from the truth")
            preds[m] = {"xhi": d["xhi"].astype(np.float32), "tb": d["tb"].astype(np.float32)}
    return z, truth, preds


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=170 if ext == "png" else None)
    plt.close(fig)
    print("wrote", name, flush=True)


# =============================================================================================
# slices
# =============================================================================================
TARGETS = (0.75, 0.5, 0.25)
cache = {c: load(c) for c in a.slice_cones}
for field, unit in (("xhi", ""), ("tb", " [mK]")):
    rows = [(c, t) for c in a.slice_cones for t in TARGETS]
    ncol = 1 + 2 * len(a.models)
    fig, axs = plt.subplots(len(rows), ncol, figsize=(2.55 * ncol + 1.2, 2.45 * len(rows)), constrained_layout=True)
    for i, (c, t) in enumerate(rows):
        z, truth, preds = cache[c]
        xbar = truth["xhi"].mean(axis=(0, 1))
        j = int(np.argmin(np.abs(xbar - t)))
        tr = truth[field][..., j]
        if field == "xhi":
            lo, hi, cmap, dlim = 0.0, 1.0, "Greys_r", 1.0
        else:
            lo, hi = np.percentile(tr, [1, 99.5])
            cmap = "magma"
            dlim = max(np.percentile(np.abs(preds[m][field][..., j] - tr), 99) for m in a.models)
        panels = [("truth", tr, cmap, lo, hi)]
        panels += [(lab(m), preds[m][field][..., j], cmap, lo, hi) for m in a.models]
        panels += [(f"{lab(m)} − truth", preds[m][field][..., j] - tr, "RdBu_r", -dlim, dlim) for m in a.models]
        for k, (title, img, cm, v0, v1) in enumerate(panels):
            ax = axs[i, k]
            im = ax.imshow(img.T, origin="lower", cmap=cm, vmin=v0, vmax=v1, extent=[0, L_PERP, 0, L_PERP])
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=8.5)
            if k >= 1:
                err = np.sqrt(np.mean((img if k > len(a.models) else img - tr) ** 2))
                ax.text(0.02, 0.97, f"RMSE {err:.3f}" if field == "xhi" else f"RMSE {err:.1f} mK",
                        transform=ax.transAxes, va="top", fontsize=7, color="w",
                        bbox=dict(facecolor="k", alpha=0.55, lw=0))
            if k in (len(a.models), ncol - 1):
                fig.colorbar(im, ax=ax, shrink=0.85)
        axs[i, 0].set_ylabel(f"cone {c} ({group_of[c]})\nz = {z[j]:.2f}, " + "$\\bar{x}_{HI}$" + f" = {xbar[j]:.2f}",
                             fontsize=8)
    name = "x$_{HI}$" if field == "xhi" else "T$_b$" + unit
    fig.suptitle(f"{name}: test lightcones (held-out blocks), emulated histories; interior 294 Mpc; "
                 "RMSE per panel (difference panels: RMSE of the difference)", fontsize=9.5)
    save(fig, f"slices_{field}")

# lightcone cuts
for c in a.slice_cones:
    z, truth, preds = cache[c]
    mid = truth["xhi"].shape[0] // 2
    rows = [("x$_{HI}$ truth", truth["xhi"][mid], "Greys_r", 0, 1)]
    rows += [(f"x$_{{HI}}$ {lab(m)}", preds[m]["xhi"][mid], "Greys_r", 0, 1) for m in a.models]
    tlo, thi = np.percentile(truth["tb"][mid], [1, 99.5])
    rows += [("T$_b$ truth [mK]", truth["tb"][mid], "magma", tlo, thi)]
    rows += [(f"T$_b$ {lab(m)}", preds[m]["tb"][mid], "magma", tlo, thi) for m in a.models]
    fig, axs = plt.subplots(len(rows), 1, figsize=(14, 1.75 * len(rows) + 0.6), sharex=True, constrained_layout=True)
    ze = np.r_[z[0] - (z[1] - z[0]) / 2, 0.5 * (z[1:] + z[:-1]), z[-1] + (z[-1] - z[-2]) / 2]
    ye = np.arange(truth["xhi"].shape[1] + 1) * (L_PERP / truth["xhi"].shape[1])
    for ax, (t, img, cm, v0, v1) in zip(axs, rows):
        im = ax.pcolormesh(ze, ye, img, cmap=cm, vmin=v0, vmax=v1, rasterized=True)
        ax.set_ylabel(t + "\n[Mpc]", fontsize=8)
        fig.colorbar(im, ax=ax, pad=0.01)
    axs[-1].set_xlabel("z (uniform-z axis resampled to the 1.4286 Mpc comoving grid)")
    fig.suptitle(f"Lightcone cut through test cone {c} ({group_of[c]}): transverse line through the centre, "
                 "every plane of the group window", fontsize=9.5)
    save(fig, f"lightcone_{c}")
del cache

# =============================================================================================
# cylindrical power spectra
# =============================================================================================
N = None
STAGE_T = {"early (x̄≈0.8)": 0.8, "mid (x̄≈0.5)": 0.5, "late (x̄≈0.2)": 0.2}
n_perp_bins, n_par_bins = 14, 10


def setup(n, nz):
    kx = 2 * np.pi * np.fft.fftfreq(n, L_PERP / n)
    kz = 2 * np.pi * np.fft.rfftfreq(nz, CELL)
    kperp = np.hypot(*np.meshgrid(kx, kx, indexing="ij"))
    perp_edges = np.geomspace(2 * np.pi / L_PERP * 0.999, np.pi / (L_PERP / n), n_perp_bins + 1)
    par_edges = np.geomspace(kz[1] * 0.999, kz[-1] * 1.001, n_par_bins + 1)
    ip = np.digitize(kperp, perp_edges) - 1
    iz = np.digitize(kz, par_edges) - 1
    ok = (ip[..., None] >= 0) & (ip[..., None] < n_perp_bins) & (iz[None, None] >= 0) & (iz[None, None] < n_par_bins)
    idx = np.where(ok, ip[..., None] * n_par_bins + iz[None, None], n_perp_bins * n_par_bins)
    k3 = np.sqrt(kperp[..., None] ** 2 + kz[None, None] ** 2) ** 3
    return idx.ravel(), k3.ravel(), perp_edges, par_edges


def binned(values, idx):
    s = np.bincount(idx, values, minlength=n_perp_bins * n_par_bins + 1)[:-1]
    c = np.bincount(idx, minlength=n_perp_bins * n_par_bins + 1)[:-1]
    return (s / np.maximum(c, 1)).reshape(n_perp_bins, n_par_bins)


cones = a.cones or sorted(group_of)
acc = {f: {s: {"truth_d2": [], **{m: {"ratio": [], "r": []} for m in a.models}, "z": [], "xbar": [], "cone": []}
           for s in STAGE_T} for f in ("xhi", "tb")}
win = np.hanning(a.chunk)
for ci, c in enumerate(cones):
    z, truth, preds = load(c)
    n, _, nz = truth["xhi"].shape
    if N is None:
        N = n
        idx, k3, perp_edges, par_edges = setup(n, a.chunk)
        norm = (L_PERP ** 2 * a.chunk * CELL) / (n * n * a.chunk) ** 2 / np.mean(win ** 2)
    xbar = truth["xhi"].mean(axis=(0, 1))
    for s, t in STAGE_T.items():
        j = int(np.argmin(np.abs(xbar - t)))
        lo = min(max(j - a.chunk // 2, 0), nz - a.chunk)
        if lo < 0 or abs(xbar[lo:lo + a.chunk].mean() - t) > 0.15:
            continue
        sl = np.s_[..., lo:lo + a.chunk]
        for f in ("xhi", "tb"):
            def ft(x):
                x = x[sl].astype(np.float64)
                x = (x - x.mean(axis=(0, 1), keepdims=True)) * win
                return np.fft.rfftn(x)
            Ft = ft(truth[f])
            Pt = np.abs(Ft) ** 2
            if Pt.sum() == 0:
                continue
            bt = binned(Pt.ravel(), idx)
            acc[f][s]["truth_d2"].append(binned((Pt * norm).ravel() * k3 / (2 * np.pi ** 2), idx))
            for m in a.models:
                Fp = ft(preds[m][f])
                bp = binned((np.abs(Fp) ** 2).ravel(), idx)
                bx = binned((Ft * Fp.conj()).real.ravel(), idx)
                acc[f][s][m]["ratio"].append(bp / bt)
                acc[f][s][m]["r"].append(bx / np.sqrt(bt * bp))
            acc[f][s]["z"].append(float(z[lo:lo + a.chunk].mean()))
            acc[f][s]["xbar"].append(float(xbar[lo:lo + a.chunk].mean()))
            acc[f][s]["cone"].append(c)
    print(f"[{ci + 1}/{len(cones)}] cone {c}", flush=True)

kp = np.sqrt(perp_edges[1:] * perp_edges[:-1])
kz = np.sqrt(par_edges[1:] * par_edges[:-1])
summary = {"k_perp_edges": perp_edges.tolist(), "k_par_edges": par_edges.tolist(), "chunk_planes": a.chunk,
           "chunk_mpc": a.chunk * CELL, "models": a.models, "fields": {}}
for f, fname, unit in (("xhi", "x$_{HI}$", ""), ("tb", "T$_b$", " [mK$^2$]")):
    stages = [s for s in STAGE_T if acc[f][s]["z"]]
    ncol = 1 + 2 * len(a.models)
    fig, axs = plt.subplots(len(stages), ncol, figsize=(3.0 * ncol + 1, 2.9 * len(stages)), constrained_layout=True,
                            squeeze=False)
    summary["fields"][f] = {}
    for i, s in enumerate(stages):
        A = acc[f][s]
        d2 = np.nanmean(np.stack(A["truth_d2"]), 0)
        maps = [("truth $\\Delta^2$" + unit, d2, "viridis", LogNorm(np.nanpercentile(d2, 2), np.nanmax(d2)))]
        maps += [(f"P / P$_{{true}}$: {lab(m)}", np.nanmedian(np.stack(A[m]["ratio"]), 0), "RdBu_r", LogNorm(0.1, 10))
                 for m in a.models]
        maps += [(f"r: {lab(m)}", np.nanmedian(np.stack(A[m]["r"]), 0), "magma", plt.Normalize(0, 1)) for m in a.models]
        summary["fields"][f][s] = {"n_chunks": len(A["z"]), "z_mean": float(np.mean(A["z"])),
                                   "xbar_mean": float(np.mean(A["xbar"])), "truth_delta2": d2.tolist(),
                                   **{m: {"ratio_median": maps[1 + k][1].tolist(),
                                          "r_median": maps[1 + len(a.models) + k][1].tolist()}
                                      for k, m in enumerate(a.models)}}
        for k, (title, M, cm, nrm) in enumerate(maps):
            ax = axs[i, k]
            im = ax.pcolormesh(perp_edges, par_edges, M.T, cmap=cm, norm=nrm, shading="flat")
            ax.set_xscale("log")
            ax.set_yscale("log")
            if k > 0 and k <= len(a.models):
                cs = ax.contour(kp, kz, M.T, levels=[0.5, 2.0], colors="k", linewidths=0.6)
                ax.clabel(cs, fontsize=6, fmt="%.1f")
            if k > len(a.models):
                cs = ax.contour(kp, kz, M.T, levels=[0.5, 0.9], colors="w", linewidths=0.6)
                ax.clabel(cs, fontsize=6, fmt="%.1f")
            if i == 0:
                ax.set_title(title)
            if i == len(stages) - 1:
                ax.set_xlabel("k$_\\perp$ [Mpc$^{-1}$]")
            if k == 0:
                ax.set_ylabel(f"{s}\n{len(A['z'])} cones, <z> = {np.mean(A['z']):.2f}\nk$_\\parallel$ [Mpc$^{{-1}}$]")
            if k in (0, len(a.models), ncol - 1):
                fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"{fname}: cylindrical power spectrum, stage-matched {a.chunk}-plane ({a.chunk * CELL:.0f} Mpc) "
                 "Hann-windowed LOS chunks, per-slice mean removed; ratio and r: median over cones "
                 "(contours: ratio 0.5/2, r 0.5/0.9)", fontsize=9.5)
    save(fig, f"cyl_{f}")
(OUT / "cyl_summary.json").write_text(json.dumps(summary))
print("wrote cyl_summary.json")
