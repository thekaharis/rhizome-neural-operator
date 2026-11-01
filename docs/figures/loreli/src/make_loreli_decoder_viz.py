"""Detailed visualization of a fine-tuned LoReLi run vs the zero-shot reference on the v3 test split.

    python make_loreli_decoder_viz.py <run_dir> <out_dir> [--stride 3] [--plot-only]

Both models get the emulated histories (v3 side file).  Computes, on every ``stride``-th test row
(plus every row of two test cones for lightcone cuts): per-row RMSE, IoU, means, transverse power
spectra and cross-correlation; random cell samples for pixel statistics; full slices for a gallery.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

sys.path.insert(0, ".")
from ebm21cm.train_loreli_finetune import load_reference
from ebm21cm.train_rhizome_mf2d import tb_normalized

W = Path("/pfs/10/work/hd_id260-fno_training")
CACHE = W / "data/compressed/loreli_slices_l294_v3.h5"
SIDE = W / "data/compressed/loreli_slices_l294_v3_emuhist.h5"
REF = W / "rhizome-neural-operator/runs/rhizome_mf2d_w64_structured_all"
CELL = 200.0 / 140
STAGES = {0: "late (x$_{HI}$ 0.05–0.35)", 1: "mid (0.35–0.65)", 2: "early (0.65–0.95)"}
NAMES = ("reference", "fine-tuned")
COL = {"truth": "k", "reference": "C3", "fine-tuned": "C0"}
plt.rcParams.update({"font.size": 9, "figure.dpi": 100})

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("out")
ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--plot-only", action="store_true")
a = ap.parse_args()
RUN, OUT = Path(a.run), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
NPZ = OUT / "decoder_viz_data.npz"

with h5py.File(CACHE, "r") as h:
    split, cone, xm, bg, z, los, stage = (h[k][:] for k in ("split", "cone_id", "xhi_mean", "background", "z",
                                                          "los_index", "stage"))
    pad, n = int(h.attrs["pad"]), int(h.attrs["n_out"])
    off, scale = json.loads(h.attrs["tb_norm"])
sl = np.s_[pad:pad + n, pad:pad + n]
test = np.flatnonzero(split == 2)
counts = {int(c): int(((cone == c) & (split == 2)).sum()) for c in np.unique(cone[test])}
strip_cones = [12316, max((c for c in counts if c != 12316), key=lambda c: counts[c])]

# --------------------------------------------------------------------------- power-spectrum bins
ky, kx = np.meshgrid(np.fft.fftfreq(n, CELL), np.fft.fftfreq(n, CELL), indexing="ij")
kk = 2 * np.pi * np.hypot(kx, ky)
kbins = np.geomspace(2 * np.pi / (n * CELL), np.pi / CELL, 17)
kidx = np.digitize(kk.ravel(), kbins) - 1
kidx[kidx < 0] = len(kbins)                                      # k = 0 mode: dropped
kc = np.sqrt(kbins[1:] * kbins[:-1])
win = np.outer(np.hanning(n), np.hanning(n))


def spectra(t, p):
    ft, fp = (np.fft.fft2((v - v.mean()) * win) for v in (t, p))
    out = []
    for q in (np.abs(ft) ** 2, np.abs(fp) ** 2, (ft * fp.conj()).real):
        out.append(np.bincount(kidx, q.ravel(), minlength=len(kbins) + 1)[:len(kc)])
    return np.array(out)                                         # (3, nk): Ptrue, Ppred, Pcross


def iou(t, p):
    T = t < 0.5
    out = []
    for P in (p < 0.5, p <= np.quantile(p, T.mean()) if 0 < T.mean() < 1 else p < 0.5):
        u = (T | P).sum()
        out.append((T & P).sum() / u if u else np.nan)
    return out                                                   # threshold 0.5, rank-matched


if not a.plot_only:
    dev = torch.device("cuda")
    models = {}
    for name, path in zip(NAMES, (REF, RUN)):
        m, _, ck = load_reference(path)
        models[name] = (m.to(dev).eval(), ck.get("scalar_keep"), tuple(ck["tb_norm"]))
    rows_stats = test[::a.stride]
    rows_strip = np.flatnonzero(np.isin(cone, strip_cones) & (split == 2))
    rows = np.union1d(rows_stats, rows_strip)
    stats_set = set(rows_stats.tolist())
    rng = np.random.default_rng(0)
    R = {k: [] for k in ("row", "x_rmse", "t_rmse", "iou05", "iou_rank", "pmean_x", "pmean_t", "tmean_t", "spec_x", "spec_t")}
    CELLS = {k: [] for k in ("stage", "tx", "tt", "rx", "rt", "fx", "ft", "rlogit", "flogit", "ru", "fu")}
    STRIP, GALLERY = {}, {}
    gal_rows = []
    for c in strip_cones:
        rr = np.flatnonzero((cone == c) & ~bg)
        gal_rows += [int(rr[np.argmin(np.abs(xm[rr] - t))]) for t in (0.75, 0.4)]
    side = h5py.File(SIDE, "r")
    with h5py.File(CACHE, "r") as h, torch.no_grad():
        for b0 in range(0, len(rows), 8):
            rb = rows[b0:b0 + 8]
            cond = torch.from_numpy(np.stack([h["cond"][r] for r in rb]).astype(np.float32)).to(dev)
            phys = torch.from_numpy(np.stack([h["phys"][r] for r in rb]).astype(np.float32)).to(dev)[:, None]
            tgt = np.stack([h["target"][r] for r in rb]).astype(np.float32)
            sc = torch.from_numpy(np.stack([side["scalars"][r] for r in rb]).astype(np.float32)).to(dev)
            res = {}
            for name, (m, keep, tbn) in models.items():
                s = sc if keep is None else sc[:, torch.as_tensor(keep, device=dev)]
                out = m(cond, s)
                tb = tb_normalized(out, phys, "structured", *tbn)
                res[name] = (out[:, 0].sigmoid().cpu().numpy(), (tb[:, 0] * scale + off).cpu().numpy(),
                             out[:, 0].cpu().numpy(), out[:, 1].cpu().numpy())
            for i, r in enumerate(rb):
                tx, tt = tgt[i, 0][sl], tgt[i, 1][sl] * scale + off
                if r in stats_set:
                    R["row"].append(r)
                    for key in ("x_rmse", "t_rmse", "iou05", "iou_rank", "pmean_x", "pmean_t", "spec_x", "spec_t"):
                        R[key].append([])
                    R["tmean_t"].append(tt.mean())
                    for name in NAMES:
                        px, pt = res[name][0][i][sl], res[name][1][i][sl]
                        R["x_rmse"][-1].append(np.sqrt(np.mean((px - tx) ** 2)))
                        R["t_rmse"][-1].append(np.sqrt(np.mean((pt - tt) ** 2)))
                        i5, ir = iou(tx, px)
                        R["iou05"][-1].append(i5); R["iou_rank"][-1].append(ir)
                        R["pmean_x"][-1].append(px.mean()); R["pmean_t"][-1].append(pt.mean())
                        R["spec_x"][-1].append(spectra(tx, px)); R["spec_t"][-1].append(spectra(tt, pt))
                    pick = rng.choice(n * n, 256, replace=False)
                    f = lambda v: v[sl].ravel()[pick]
                    CELLS["stage"].append(np.full(256, stage[r]))
                    CELLS["tx"].append(tx.ravel()[pick]); CELLS["tt"].append(tt.ravel()[pick])
                    for tag, name in (("r", "reference"), ("f", "fine-tuned")):
                        CELLS[tag + "x"].append(f(res[name][0][i])); CELLS[tag + "t"].append(f(res[name][1][i]))
                    CELLS["rlogit"].append(f(res["reference"][2][i])); CELLS["flogit"].append(f(res["fine-tuned"][2][i]))
                    CELLS["ru"].append(f(res["reference"][3][i])); CELLS["fu"].append(f(res["fine-tuned"][3][i]))
                if cone[r] in strip_cones:
                    mid = n // 2
                    STRIP.setdefault(int(cone[r]), []).append(
                        [los[r], z[r], tx[mid], tt[mid]] + [res[nm][j][i][sl][mid] for nm in NAMES for j in (0, 1)])
                if r in gal_rows:
                    GALLERY[int(r)] = np.stack([tx, tt] + [res[nm][j][i][sl] for nm in NAMES for j in (0, 1)])
            if b0 % 800 == 0:
                print(f"{b0}/{len(rows)}", flush=True)
    save = {k: np.asarray(v) for k, v in R.items()}
    save.update({f"cells_{k}": np.concatenate(v) for k, v in CELLS.items()})
    for c, v in STRIP.items():
        v = sorted(v, key=lambda e: e[0])
        save[f"strip_{c}_los"] = np.array([e[0] for e in v]); save[f"strip_{c}_z"] = np.array([e[1] for e in v])
        save[f"strip_{c}_img"] = np.array([np.stack(e[2:]) for e in v])          # (rows, 6, n)
    for r, v in GALLERY.items():
        save[f"gal_{r}"] = v
    save["kc"] = kc
    np.savez_compressed(NPZ, **save)

# =========================================================================== plots
D = np.load(NPZ)
rows = D["row"]
st, zz = stage[rows], z[rows]
def saveall(fig, name):
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{e}", dpi=170 if e == "png" else None)
    plt.close(fig)
summary = {"run": str(RUN), "rows": int(len(rows)), "stride": a.stride, "stages": {}}
for s, lab in [(-1, "outside stage bins")] + list(STAGES.items()):
    m = st == s
    if m.any():
        summary["stages"][lab] = {nm: {"x_rmse": float(np.sqrt(np.mean(D["x_rmse"][m, j] ** 2))),
                                       "t_rmse_mK": float(np.sqrt(np.mean(D["t_rmse"][m, j] ** 2))),
                                       "iou05": float(np.nanmean(D["iou05"][m, j])), "iou_rank": float(np.nanmean(D["iou_rank"][m, j])),
                                       "mean_x_bias": float(np.mean(D["pmean_x"][m, j] - xm[rows][m])),
                                       "mean_t_bias_mK": float(np.mean(D["pmean_t"][m, j] - D["tmean_t"][m]))}
                                  for j, nm in enumerate(NAMES)} | {"rows": int(m.sum())}
json.dump(summary, open(OUT / "decoder_viz_summary.json", "w"), indent=1)

# 1) gallery
gal = sorted([k for k in D.files if k.startswith("gal_")], key=lambda k: (cone[int(k[4:])], -xm[int(k[4:])]))
fig, ax = plt.subplots(len(gal), 8, figsize=(21, 2.75 * len(gal)), constrained_layout=True)
titles = ["x$_{HI}$ truth", "x$_{HI}$ reference", "x$_{HI}$ fine-tuned", "x$_{HI}$ fine-tuned − truth",
          "T$_b$ truth [mK]", "T$_b$ reference", "T$_b$ fine-tuned", "T$_b$ fine-tuned − truth [mK]"]
for i, k in enumerate(gal):
    r = int(k[4:]); g = D[k]                                     # tx, tt, rx, rt, fx, ft
    lo, hi = np.percentile(g[1], [1, 99.5]); dt = np.percentile(np.abs(g[5] - g[1]), 99)
    panels = [(g[0], "Greys_r", 0, 1), (g[2], "Greys_r", 0, 1), (g[4], "Greys_r", 0, 1), (g[4] - g[0], "RdBu_r", -1, 1),
              (g[1], "magma", lo, hi), (g[3], "magma", lo, hi), (g[5], "magma", lo, hi), (g[5] - g[1], "RdBu_r", -dt, dt)]
    for j, (img, cm, v0, v1) in enumerate(panels):
        im = ax[i, j].imshow(img, cmap=cm, vmin=v0, vmax=v1, origin="lower")
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if i == 0:
            ax[i, j].set_title(titles[j], fontsize=9)
        if j in (1, 2, 5, 6):
            truth = g[0] if j < 4 else g[1]
            e = np.sqrt(np.mean((img - truth) ** 2))
            ax[i, j].text(0.02, 0.97, f"{e:.3f}" if j < 4 else f"{e:.1f} mK", transform=ax[i, j].transAxes, va="top",
                          fontsize=8, color="w", bbox=dict(facecolor="k", alpha=0.55, lw=0))
        if j in (2, 3, 6, 7):
            fig.colorbar(im, ax=ax[i, j], shrink=0.8)
    ax[i, 0].set_ylabel(f"cone {cone[r]}\nz = {z[r]:.2f}, <x$_{{HI}}$> = {xm[r]:.2f}", fontsize=9)
fig.suptitle("Test cones (held-out blocks), emulated histories; numbers are slice RMSE. Interior 206×206 cells = 294 Mpc.", fontsize=10)
saveall(fig, "dec_gallery")

# 2) lightcone cuts
for c in strip_cones:
    img, zs, ls_ = D[f"strip_{c}_img"], D[f"strip_{c}_z"], D[f"strip_{c}_los"]
    keep = np.r_[True, np.cumsum(np.diff(ls_) > 8) == 0]          # contiguous run; drop isolated far background rows
    img, zs = img[keep], zs[keep]
    fig, ax = plt.subplots(6, 1, figsize=(14, 10.5), sharex=True, constrained_layout=True)
    ze = np.r_[zs[0] - (zs[1] - zs[0]) / 2, 0.5 * (zs[1:] + zs[:-1]), zs[-1] + (zs[-1] - zs[-2]) / 2]
    ye = np.arange(n + 1) * CELL
    tlo, thi = np.percentile(img[:, 1], [1, 99.5])
    labels = ["x$_{HI}$ truth", "x$_{HI}$ reference", "x$_{HI}$ fine-tuned", "T$_b$ truth [mK]", "T$_b$ reference", "T$_b$ fine-tuned"]
    order = [0, 2, 4, 1, 3, 5]
    for p, (k, lab) in enumerate(zip(order, labels)):
        x_like = p < 3
        im = ax[p].pcolormesh(ze, ye, img[:, k].T, cmap="Greys_r" if x_like else "magma",
                              vmin=0 if x_like else tlo, vmax=1 if x_like else thi, rasterized=True)
        ax[p].set_ylabel(lab + "\n[Mpc]", fontsize=8)
        fig.colorbar(im, ax=ax[p], pad=0.01)
    ax[-1].set_xlabel("z (true, uniform-z axis); one column per cache row = one LOS plane")
    fig.suptitle(f"Lightcone cut through test cone {c} (transverse line through the slice centre, all rows of the cone)", fontsize=10)
    saveall(fig, f"dec_lightcone_{c}")

# 3) pixel statistics
C = {k[6:]: D[k] for k in D.files if k.startswith("cells_")}
fig, ax = plt.subplots(2, 4, figsize=(17, 7.6), constrained_layout=True)
for j, (tag, nm) in enumerate((("r", "reference"), ("f", "fine-tuned"))):
    ax[0, j].hist2d(C["tx"], C[tag + "x"], bins=60, range=[[0, 1], [0, 1]], norm=LogNorm(), cmap="viridis")
    ax[0, j].plot([0, 1], [0, 1], "r-", lw=0.6); ax[0, j].set_title(f"x$_{{HI}}$ per cell: {nm}")
    ax[0, j].set_xlabel("truth"); ax[0, j].set_ylabel("prediction")
    lo, hi = np.percentile(C["tt"], [0.5, 99.5])
    ax[1, j].hist2d(C["tt"], C[tag + "t"], bins=60, range=[[lo, hi], [lo, hi]], norm=LogNorm(), cmap="viridis")
    ax[1, j].plot([lo, hi], [lo, hi], "r-", lw=0.6); ax[1, j].set_title(f"T$_b$ per cell [mK]: {nm}")
    ax[1, j].set_xlabel("truth"); ax[1, j].set_ylabel("prediction")
for s, ls in ((0, ":"), (1, "--"), (2, "-")):
    m = C["stage"] == s
    for key, axx, rng_ in (("x", ax[0, 2], (0, 1)), ("t", ax[1, 2], np.percentile(C["tt"], [0.5, 99.5]))):
        for src, nm in (("t", "truth"), ("r", "reference"), ("f", "fine-tuned")):
            v = C["t" + key] if src == "t" else C[src + key]
            axx.hist(v[m], bins=60, range=tuple(rng_), histtype="step", density=True, color=COL[nm], ls=ls,
                     label=f"{nm}, {STAGES[s].split(' ')[0]}" if key == "x" else None)
ax[0, 2].set_yscale("log"); ax[0, 2].set_title("x$_{HI}$ PDF by stage"); ax[0, 2].legend(fontsize=6, ncol=1)
ax[1, 2].set_title("T$_b$ PDF by stage [mK]")
# calibration: mean truth in bins of predicted x
for tag, nm in (("r", "reference"), ("f", "fine-tuned")):
    bins = np.linspace(0, 1, 21); idx = np.digitize(C[tag + "x"], bins) - 1
    mt = [C["tx"][idx == b].mean() if (idx == b).sum() > 50 else np.nan for b in range(20)]
    ax[0, 3].plot(0.5 * (bins[1:] + bins[:-1]), mt, "o-", color=COL[nm], ms=3, label=nm)
ax[0, 3].plot([0, 1], [0, 1], "k:", lw=0.8); ax[0, 3].set_title("calibration: mean true x$_{HI}$ per predicted bin")
ax[0, 3].set_xlabel("predicted x$_{HI}$"); ax[0, 3].set_ylabel("mean true x$_{HI}$"); ax[0, 3].legend(fontsize=7)
for tag, nm in (("r", "reference"), ("f", "fine-tuned")):
    lo, hi = np.percentile(C["tt"], [0.5, 99.5]); bins = np.linspace(lo, hi, 25); idx = np.digitize(C[tag + "t"], bins) - 1
    mt = [C["tt"][idx == b].mean() if (idx == b).sum() > 50 else np.nan for b in range(24)]
    ax[1, 3].plot(0.5 * (bins[1:] + bins[:-1]), mt, "o-", color=COL[nm], ms=3, label=nm)
ax[1, 3].plot([lo, hi], [lo, hi], "k:", lw=0.8); ax[1, 3].set_title("calibration: mean true T$_b$ per predicted bin")
ax[1, 3].set_xlabel("predicted T$_b$ [mK]"); ax[1, 3].legend(fontsize=7)
fig.suptitle(f"Cell statistics: {len(C['tx']):,} random interior cells from {len(rows):,} test rows", fontsize=10)
saveall(fig, "dec_pixel_stats")

# 4) metrics along the light cone
fig, ax = plt.subplots(2, 3, figsize=(16, 7.8), constrained_layout=True)
zb = np.linspace(zz[st >= 0].min(), zz[st >= 0].max() + 1e-6, 15); zi = np.digitize(zz, zb) - 1; zcen = 0.5 * (zb[1:] + zb[:-1])
active = st >= 0
for j, nm in enumerate(NAMES):
    for axx, key, lab in ((ax[0, 0], "x_rmse", "x$_{HI}$ RMSE"), (ax[0, 1], "t_rmse", "T$_b$ RMSE [mK]")):
        med = [np.sqrt(np.mean(D[key][(zi == b) & active, j] ** 2)) if ((zi == b) & active).sum() > 5 else np.nan for b in range(len(zcen))]
        axx.plot(zcen, med, "o-", color=COL[nm], ms=3, label=nm); axx.set_title(lab + " vs z (active rows)"); axx.set_xlabel("z")
    xb = np.linspace(0, 1, 21); xi = np.digitize(xm[rows], xb) - 1
    for key, ls, lab in (("iou05", "-", "IoU (x<0.5)"), ("iou_rank", "--", "IoU (rank-matched)")):
        v = [np.nanmean(D[key][xi == b, j]) if (xi == b).sum() > 5 else np.nan for b in range(20)]
        ax[0, 2].plot(0.5 * (xb[1:] + xb[:-1]), v, ls, color=COL[nm], label=f"{nm}: {lab}")
    ax[1, 0].scatter(xm[rows], D["pmean_x"][:, j], s=2, alpha=0.3, color=COL[nm], label=nm)
    ax[1, 1].scatter(D["tmean_t"], D["pmean_t"][:, j], s=2, alpha=0.3, color=COL[nm], label=nm)
ax[0, 0].legend(fontsize=7); ax[0, 2].legend(fontsize=6); ax[0, 2].set_xlabel("slice-mean x$_{HI}$ (truth)")
ax[0, 2].set_title("ionized-region IoU vs stage")
ax[1, 0].plot([0, 1], [0, 1], "k:"); ax[1, 0].set_xlabel("true slice-mean x$_{HI}$"); ax[1, 0].set_ylabel("predicted slice mean")
ax[1, 0].set_title("slice-mean x$_{HI}$ (does the map follow the history?)"); ax[1, 0].legend(fontsize=7, markerscale=4)
lo, hi = np.percentile(D["tmean_t"], [0.5, 99.5]); ax[1, 1].plot([lo, hi], [lo, hi], "k:")
ax[1, 1].set_xlabel("true slice-mean T$_b$ [mK]"); ax[1, 1].set_ylabel("predicted"); ax[1, 1].set_title("slice-mean T$_b$")
gain_x = D["x_rmse"][:, 0] - D["x_rmse"][:, 1]; gain_t = D["t_rmse"][:, 0] - D["t_rmse"][:, 1]
ax[1, 2].hist(gain_x / np.maximum(D["x_rmse"][:, 0], 1e-6), bins=60, range=(-1, 1), histtype="step", label="x$_{HI}$")
ax[1, 2].hist(gain_t / np.maximum(D["t_rmse"][:, 0], 1e-6), bins=60, range=(-1, 1), histtype="step", label="T$_b$")
ax[1, 2].axvline(0, color="k", lw=0.6); ax[1, 2].set_xlabel("relative RMSE reduction per row (>0: fine-tuned better)")
ax[1, 2].set_title(f"per-row gain: better on {np.mean(gain_x > 0):.0%} (x$_{{HI}}$) / {np.mean(gain_t > 0):.0%} (T$_b$) of rows")
ax[1, 2].legend(fontsize=7)
saveall(fig, "dec_metrics")

# 5) power spectra
kc_ = D["kc"]
fig, ax = plt.subplots(2, 3, figsize=(15, 7.2), constrained_layout=True, sharex=True)
for col, s in enumerate((0, 1, 2)):
    m = st == s
    for row_, key, lab in ((0, "spec_x", "x$_{HI}$"), (1, "spec_t", "T$_b$")):
        S = D[key][m]                                          # (rows, model, 3, nk)
        for j, nm in enumerate(NAMES):
            pt, pp, px = (S[:, j, q].mean(0) for q in range(3))
            ax[row_, col].plot(kc_, pp / pt, "-", color=COL[nm], label=f"{nm}: P$_{{pred}}$/P$_{{true}}$")
            ax[row_, col].plot(kc_, px / np.sqrt(pt * pp), "--", color=COL[nm], label=f"{nm}: r(k)")
        ax[row_, col].axhline(1, color="k", lw=0.5); ax[row_, col].set_xscale("log"); ax[row_, col].set_ylim(0, 1.6)
        ax[row_, col].set_title(f"{lab}, {STAGES[s]}", fontsize=9)
        if row_ == 1:
            ax[row_, col].set_xlabel("k [Mpc$^{-1}$]")
ax[0, 0].legend(fontsize=6)
fig.suptitle("Transverse power-spectrum ratio and cross-correlation r(k) with the truth (Hann-windowed interior, mean over test rows)", fontsize=10)
saveall(fig, "dec_power")

# 6) what the decoder changed
ref, _, _ = load_reference(REF); ft_, _, _ = load_reference(RUN)
pr = dict(ref.named_parameters()); pf = dict(ft_.named_parameters())
fig, ax = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
names = [k for k in pf if k.startswith("decode")]
ax[0].bar(range(len(names)), [float((pf[k] - pr[k]).norm() / pr[k].norm()) for k in names])
ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=30, fontsize=7)
ax[0].set_title("relative weight change ‖Δw‖/‖w‖\n(all other 18.5 M weights unchanged)", fontsize=9)
ax[1].hist2d(C["rx"], C["fx"], bins=60, range=[[0, 1], [0, 1]], norm=LogNorm(), cmap="viridis")
ax[1].plot([0, 1], [0, 1], "r-", lw=0.6); ax[1].set_xlabel("reference x$_{HI}$"); ax[1].set_ylabel("fine-tuned x$_{HI}$")
ax[1].set_title("same cell, same features: the learned remap of x$_{HI}$", fontsize=9)
lo, hi = np.percentile(np.r_[C["rlogit"], C["flogit"]], [0.5, 99.5])
ax[2].hist2d(C["rlogit"], C["flogit"], bins=60, range=[[lo, hi], [lo, hi]], norm=LogNorm(), cmap="viridis")
ax[2].plot([lo, hi], [lo, hi], "r-", lw=0.6); ax[2].set_xlabel("reference logit"); ax[2].set_ylabel("fine-tuned logit")
ax[2].set_title(f"x$_{{HI}}$ logit: spread {np.std(C['rlogit']):.1f} → {np.std(C['flogit']):.1f}", fontsize=9)
lo, hi = np.percentile(C["tt"], [0.5, 99.5])
ax[3].hist2d(C["rt"], C["ft"], bins=60, range=[[lo, hi], [lo, hi]], norm=LogNorm(), cmap="viridis")
ax[3].plot([lo, hi], [lo, hi], "r-", lw=0.6); ax[3].set_xlabel("reference T$_b$ [mK]"); ax[3].set_ylabel("fine-tuned T$_b$ [mK]")
ax[3].set_title("learned remap of T$_b$", fontsize=9)
saveall(fig, "dec_decoder_change")
print(json.dumps(summary, indent=1))
