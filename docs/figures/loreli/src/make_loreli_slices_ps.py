"""Representative test slices and 2-D (cylindrical) power spectra: truth vs reference vs a fine-tuned arm.

    python docs/figures/loreli/src/make_loreli_slices_ps.py [--models reference b8_pointwise] [--out docs/figures/loreli/slices_ps]

Inputs are the full test lightcones used by 21cmEVAL (no inference here):
    truth   data/eval_cubes/loreli_l294_v3_test/<group>/cone<N>.npz   (xhi, tb [mK], z_grid)
    models  runs/cm21eval/preds/<model>/cone<N>.npz                    (xhi, tb; float16)
Geometry: 206 x 206 transverse cells of 294/206 = 1.427 Mpc (periodic), LOS planes on the uniform
1.4286 Mpc comoving grid of the LoReLi adapter.

Figures
    slices_xhi / slices_tb   six representative slices (cone, stage): truth, models, model - truth
    slices_pk                the same slices: transverse Delta^2(k), P_model/P_true and r(k)
    cyl_ps_xhi / cyl_ps_tb   cylindrical Delta^2(k_perp, k_par) of the truth and, per model, the ratio
                             to the truth and the cross-correlation r(k_perp, k_par), per stage;
                             chunks of 128 LOS planes (183 Mpc, Blackman-Harris taper along the LOS)
                             every 32 planes, assigned to the stage of their mean true x_HI, averaged
                             over all chunks of all 30 test cones.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np

REPO = Path(__file__).resolve().parents[4]
W = REPO.parent
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["reference", "b8_pointwise"])
ap.add_argument("--truth", default=str(W / "data/eval_cubes/loreli_l294_v3_test"))
ap.add_argument("--preds", default=str(REPO / "runs/cm21eval/preds"))
ap.add_argument("--groups", default=str(REPO / "runs/cm21eval/groups.json"))
ap.add_argument("--out", default=str(REPO / "docs/figures/loreli/slices_ps"))
ap.add_argument("--chunk", type=int, default=128)
ap.add_argument("--step", type=int, default=32)
a = ap.parse_args()
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
groups = json.loads(Path(a.groups).read_text())
group_of = {c: g for g, v in groups.items() for c in v["cones"]}
MODELS = a.models
LABEL = {"reference": "reference (zero-shot)", "reference_oracle": "reference, oracle hist."}
label = lambda m: LABEL.get(m, m.replace("b8_", "8 blk: ").replace("b44_", "44 blk: ").replace("_oracle", ", oracle hist."))
COLORS = ["0.2", "#08519c", "#de2d26", "#31a354"]
CELL_T = 294.0 / 206
CELL_L = 200.0 / 140
STAGES = [("late", 0.05, 0.35), ("mid", 0.35, 0.65), ("early", 0.65, 0.95)]
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "legend.fontsize": 7})


def load(cone):
    g = group_of[cone]
    with np.load(Path(a.truth) / g / f"cone{cone}.npz") as d:
        truth = {"xhi": d["xhi"].astype(np.float32), "tb": d["tb"].astype(np.float32), "z": d["z_grid"]}
    preds = {}
    for m in MODELS:
        with np.load(Path(a.preds) / m / f"cone{cone}.npz") as d:
            preds[m] = {"xhi": d["xhi"].astype(np.float32), "tb": d["tb"].astype(np.float32)}
    return truth, preds


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=170 if ext == "png" else None)
    plt.close(fig)
    print("wrote", name, flush=True)


# ============================================================================================ slices
PICK = [(12316, 0.85), (12316, 0.50), (12316, 0.15), (12930, 0.50), (19547, 0.30), (13288, 0.90)]
n = 206
kx = 2 * np.pi * np.fft.fftfreq(n, CELL_T)
kperp2d = np.hypot(*np.meshgrid(kx, kx, indexing="ij"))
kb = np.geomspace(2 * np.pi / (n * CELL_T), np.pi / CELL_T, 21)
kc = np.sqrt(kb[1:] * kb[:-1])
kid = np.digitize(kperp2d.ravel(), kb) - 1
kid[(kid < 0) | (kid >= len(kc))] = len(kc)
area = (n * CELL_T) ** 2


def pk2d(u, v=None):
    """Transverse (periodic) P(k) [field^2 Mpc^2] of one slice; cross spectrum if v is given."""
    fu = np.fft.fft2(u - u.mean())
    fv = fu if v is None else np.fft.fft2(v - v.mean())
    p = (fu * fv.conj()).real * CELL_T ** 4 / area
    s = np.bincount(kid, p.ravel(), minlength=len(kc) + 1)[:len(kc)]
    c = np.bincount(kid, minlength=len(kc) + 1)[:len(kc)]
    return s / np.maximum(c, 1)


rows = []
for cone, target in PICK:
    truth, preds = load(cone)
    xbar = truth["xhi"].mean((0, 1))
    j = int(np.argmin(np.abs(xbar - target)))
    rows.append({"cone": cone, "j": j, "z": float(truth["z"][j]), "xbar": float(xbar[j]), "group": group_of[cone],
                 "truth": {f: truth[f][..., j] for f in ("xhi", "tb")},
                 "pred": {m: {f: preds[m][f][..., j] for f in ("xhi", "tb")} for m in MODELS}})
    del truth, preds

for field, fname, unit in (("xhi", "x$_{HI}$", ""), ("tb", "T$_b$", " [mK]")):
    ncol = 1 + 2 * len(MODELS)
    fig, axs = plt.subplots(len(rows), ncol, figsize=(3.05 * ncol, 2.85 * len(rows)), constrained_layout=True)
    for i, r in enumerate(rows):
        t = r["truth"][field]
        lo, hi = (0, 1) if field == "xhi" else tuple(np.percentile(t, [1, 99.5]))
        dmax = 1.0 if field == "xhi" else float(max(np.percentile(np.abs(r["pred"][m][field] - t), 99) for m in MODELS))
        panels = [(t, f"{fname} truth{unit}", "Greys_r" if field == "xhi" else "magma", lo, hi, None)]
        panels += [(r["pred"][m][field], label(m), "Greys_r" if field == "xhi" else "magma", lo, hi, t) for m in MODELS]
        panels += [(r["pred"][m][field] - t, f"{label(m)} − truth", "RdBu_r", -dmax, dmax, None) for m in MODELS]
        for jcol, (img, title, cmap, v0, v1, ref) in enumerate(panels):
            ax = axs[i, jcol]
            im = ax.imshow(img.T, origin="lower", cmap=cmap, vmin=v0, vmax=v1, extent=[0, n * CELL_T, 0, n * CELL_T])
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title)
            if ref is not None:
                e = float(np.sqrt(np.mean((img - ref) ** 2)))
                ax.text(0.02, 0.97, f"RMSE {e:.3f}" if field == "xhi" else f"RMSE {e:.1f} mK", transform=ax.transAxes,
                        va="top", fontsize=7.5, color="w", bbox=dict(facecolor="k", alpha=0.55, lw=0))
            if jcol in (len(MODELS), ncol - 1):
                fig.colorbar(im, ax=ax, shrink=0.85)
        axs[i, 0].set_ylabel(f"cone {r['cone']} ({r['group'].split('_')[0]})\nz = {r['z']:.2f}, "
                             f"$\\bar{{x}}_{{HI}}$ = {r['xbar']:.2f}", fontsize=8.5)
    fig.suptitle(f"{fname}: representative LoReLi II test slices (held-out blocks, emulated histories); 294 Mpc, "
                 "206² cells", fontsize=10)
    save(fig, f"slices_{field}")

fig, axs = plt.subplots(len(rows), 4, figsize=(15, 2.7 * len(rows)), constrained_layout=True, sharex=True)
for i, r in enumerate(rows):
    for jf, (field, fname, unit) in enumerate((("xhi", "x$_{HI}$", ""), ("tb", "T$_b$", " [mK$^2$]"))):
        t = r["truth"][field]
        pt = pk2d(t)
        d2 = lambda p: kc ** 2 * p / (2 * np.pi)                  # 2-D dimensionless power
        ax, axr = axs[i, 2 * jf], axs[i, 2 * jf + 1]
        ax.loglog(kc, d2(pt), "k", lw=2, label="truth")
        for m, c in zip(MODELS, COLORS):
            pp = pk2d(r["pred"][m][field])
            ax.loglog(kc, d2(pp), color=c, lw=1.3, label=label(m))
            axr.plot(kc, pp / pt, color=c, lw=1.3)
            axr.plot(kc, pk2d(r["pred"][m][field], t) / np.sqrt(pp * pt), color=c, lw=1.3, ls="--")
        axr.axhline(1, color="k", lw=0.6)
        axr.set_xscale("log")
        axr.set_ylim(0, 1.6)
        ax.set_ylabel(f"Δ²$_{{2D}}$({fname}){unit}")
        if i == 0:
            ax.set_title(f"{fname}: Δ²(k) = k² P(k) / 2π")
            axr.set_title(f"{fname}: P$_{{model}}$/P$_{{true}}$ (solid), r(k) (dashed)")
        axs[i, 0].text(0.02, 0.04, f"cone {r['cone']}, z = {r['z']:.2f}, $\\bar{{x}}_{{HI}}$ = {r['xbar']:.2f}",
                       transform=axs[i, 0].transAxes, fontsize=7.5)
for ax in axs[-1]:
    ax.set_xlabel("k$_\\perp$ [Mpc$^{-1}$]")
axs[0, 0].legend(loc="lower right")
fig.suptitle("Transverse power spectra of the representative slices", fontsize=10)
save(fig, "slices_pk")

# ============================================================================================ cylindrical P(k_perp, k_par)
L = a.chunk
taper = np.blackman(L).astype(np.float32)        # LOS is not periodic
kpar = np.abs(2 * np.pi * np.fft.fftfreq(L, CELL_L))
kpb = np.geomspace(2 * np.pi / (n * CELL_T), np.pi / CELL_T, 13)
klb = np.geomspace(2 * np.pi / (L * CELL_L), np.pi / CELL_L, 11)
ip = np.digitize(kperp2d, kpb) - 1                # (n, n)
il = np.digitize(kpar, klb) - 1                   # (L,)
ok = (ip[..., None] >= 0) & (ip[..., None] < len(kpb) - 1) & (il[None, None, :] >= 0) & (il[None, None, :] < len(klb) - 1)
flat = np.where(ok, ip[..., None] * (len(klb) - 1) + il[None, None, :], -1).ravel()
nb = (len(kpb) - 1) * (len(klb) - 1)
counts = np.bincount(flat[flat >= 0], minlength=nb)
vol = (n * CELL_T) ** 2 * L * CELL_L * np.mean(taper ** 2)


def cyl(u, v=None):
    fu = np.fft.fftn((u - u.mean((0, 1), keepdims=True)) * taper)
    fv = fu if v is None else np.fft.fftn((v - v.mean((0, 1), keepdims=True)) * taper)
    p = ((fu * fv.conj()).real * (CELL_T ** 2 * CELL_L) ** 2 / vol).ravel()
    s = np.bincount(flat[flat >= 0], p[flat >= 0], minlength=nb)
    return (s / np.maximum(counts, 1)).reshape(len(kpb) - 1, len(klb) - 1)


acc = {s[0]: {"n": 0, **{f: {"tt": 0, **{m: {"pp": 0, "pt": 0} for m in MODELS}} for f in ("xhi", "tb")}} for s in STAGES}
for gi, (g, v) in enumerate(groups.items()):
    for cone in v["cones"]:
        truth, preds = load(cone)
        nz = truth["xhi"].shape[2]
        for s0 in range(0, nz - L + 1, a.step):
            sl = slice(s0, s0 + L)
            xm = float(truth["xhi"][..., sl].mean())
            st = next((s for s, lo, hi in STAGES if lo <= xm < hi), None)
            if st is None:
                continue
            acc[st]["n"] += 1
            for f in ("xhi", "tb"):
                t = truth[f][..., sl]
                acc[st][f]["tt"] = acc[st][f]["tt"] + cyl(t)
                for m in MODELS:
                    p = preds[m][f][..., sl]
                    acc[st][f][m]["pp"] = acc[st][f][m]["pp"] + cyl(p)
                    acc[st][f][m]["pt"] = acc[st][f][m]["pt"] + cyl(p, t)
        print(f"cone {cone}: done", flush=True)
        del truth, preds

kpc, klc = np.sqrt(kpb[1:] * kpb[:-1]), np.sqrt(klb[1:] * klb[:-1])
K = np.hypot(kpc[:, None], klc[None, :])
summary = {"chunk_planes": L, "step": a.step, "k_perp_edges": kpb.tolist(), "k_par_edges": klb.tolist(), "stages": {}}
for f, fname, unit in (("xhi", "x$_{HI}$", ""), ("tb", "T$_b$", " [mK$^2$]")):
    ncol = 1 + 2 * len(MODELS)
    fig, axs = plt.subplots(len(STAGES), ncol, figsize=(3.5 * ncol, 3.1 * len(STAGES)), constrained_layout=True)
    for i, (st, lo, hi) in enumerate(STAGES):
        nch = acc[st]["n"]
        tt = acc[st][f]["tt"] / max(nch, 1)
        d2 = K ** 3 * tt / (2 * np.pi ** 2)
        mesh = lambda ax, Z, **kw: ax.pcolormesh(kpb, klb, Z.T, shading="flat", **kw)
        im = mesh(axs[i, 0], d2, norm=LogNorm(vmin=np.nanmax(d2) * 1e-3, vmax=np.nanmax(d2)), cmap="viridis")
        fig.colorbar(im, ax=axs[i, 0], shrink=0.85)
        axs[i, 0].set_ylabel(f"{st} ($\\bar{{x}}_{{HI}}$ {lo}–{hi}, {nch} chunks)\nk$_\\parallel$ [Mpc$^{{-1}}$]")
        summary["stages"].setdefault(st, {})["chunks"] = nch
        for jm, m in enumerate(MODELS):
            pp = acc[st][f][m]["pp"] / max(nch, 1)
            pt = acc[st][f][m]["pt"] / max(nch, 1)
            ratio, r = pp / tt, pt / np.sqrt(pp * tt)
            im1 = mesh(axs[i, 1 + jm], ratio, norm=LogNorm(vmin=0.1, vmax=10), cmap="RdBu_r")
            im2 = mesh(axs[i, 1 + len(MODELS) + jm], r, vmin=0, vmax=1, cmap="magma")
            if jm == len(MODELS) - 1:
                fig.colorbar(im1, ax=axs[i, 1 + jm], shrink=0.85)
                fig.colorbar(im2, ax=axs[i, ncol - 1], shrink=0.85)
            summary["stages"][st][f"{f}_{m}"] = {"ratio": ratio.tolist(), "r": r.tolist()}
            if i == 0:
                axs[i, 1 + jm].set_title(f"P$_{{model}}$ / P$_{{true}}$: {label(m)}")
                axs[i, 1 + len(MODELS) + jm].set_title(f"r(k$_\\perp$, k$_\\parallel$): {label(m)}")
        summary["stages"][st][f"{f}_truth_delta2"] = d2.tolist()
        if i == 0:
            axs[i, 0].set_title(f"truth: Δ²({fname}){unit}")
    for ax in axs.ravel():
        ax.set_xscale("log")
        ax.set_yscale("log")
    for ax in axs[-1]:
        ax.set_xlabel("k$_\\perp$ [Mpc$^{-1}$]")
    fig.suptitle(f"{fname}: cylindrical power spectrum P(k$_\\perp$, k$_\\parallel$) of the 30 test lightcones, chunks of {L} "
                 f"planes ({L * CELL_L:.0f} Mpc) every {a.step}, Blackman-Harris taper along the LOS; stage = chunk-mean "
                 "true x$_{HI}$", fontsize=9.5)
    save(fig, f"cyl_ps_{f}")
(OUT / "cyl_ps.json").write_text(json.dumps(summary))
(OUT / "slices.json").write_text(json.dumps([{k: r[k] for k in ("cone", "j", "z", "xbar", "group")} for r in rows], indent=1))
print("done")
