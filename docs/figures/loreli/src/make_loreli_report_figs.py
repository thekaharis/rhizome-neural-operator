"""Figures for the LoReLi integration report.  Run from the rhizome-loreli worktree with fno-env."""
import glob
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from ebm21cm.cosmology import FlatLCDM
from ebm21cm.data.loreli import LoreliCone


def comoving_axis(n=1536, z0=5.0, z1=15.0, omega_m=0.308, h=0.678):
    """The axis the v2 pipeline assumed: uniform in comoving distance between z0 and z1."""
    cosmo = FlatLCDM(omega_m, h)
    chi = np.linspace(cosmo.comoving_distance(z0), cosmo.comoving_distance(z1), n)
    return cosmo.z_at_distance(chi), chi - chi[0]

W = Path("/pfs/10/work/hd_id260-fno_training")
R = Path(__file__).resolve().parents[4]                      # repository root
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "docs/figures/loreli"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)

G = h5py.File(W / "data/compressed/loreli_globals_v1.h5", "r")
CACHE = W / "data/compressed/loreli_slices_l294_v2.h5"
SIDE = W / "data/compressed/loreli_slices_l294_v2_emuhist.h5"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
summary = {}

sim, kind = G["sim"][:], G["lightcone"][:]
five = np.flatnonzero(kind == 2)
diag, x5 = G["diag"][:], G["xhi5_native"][:]
zu = G["z_native"][:]                       # true: uniform in z
zc, dc = comoving_axis()                    # assumed in v2: uniform comoving (loreli_redshifts before the fix)
nu = lambda z: 1420.405751 / (1 + z)
zf = 1420.405751 / np.linspace(nu(5), nu(15), 1536) - 1
idx = np.arange(1536)

# ---------------- Fig: LOS geometry
est = [[] for _ in range(1536)]
for i in five:
    d = diag[i]; ok = np.isfinite(d[0]); zb, xb = d[0][ok], 1 - d[1][ok]
    xm = np.maximum.accumulate(xb)
    y = x5[i]
    for k in np.flatnonzero(np.isfinite(y) & (y > 0.1) & (y < 0.9)):
        if xm.min() < y[k] < xm.max():
            est[k].append(np.interp(y[k], xm, zb))
ks = [k for k in range(0, 1536, 8) if len(est[k]) >= 5]
med = np.array([np.median(est[k]) for k in ks]); q1 = np.array([np.percentile(est[k], 16) for k in ks])
q2 = np.array([np.percentile(est[k], 84) for k in ks])
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
ax[0].fill_between(ks, q1, q2, color="k", alpha=0.15, lw=0, label="16-84% over cones")
ax[0].plot(ks, med, "k.", ms=3, label="z inferred from box x$_{HI}$(z)")
for g, lab, c in ((zu, "uniform in z", "C2"), (zc, "uniform comoving (assumed)", "C3"), (zf, "uniform frequency", "C0")):
    ax[0].plot(idx, g, c, lw=1.2, label=lab)
ax[0].set_xlim(0, 700); ax[0].set_ylim(4.8, 9.5); ax[0].set_xlabel("LOS plane index"); ax[0].set_ylabel("z")
ax[0].set_title("(a) redshift of each plane"); ax[0].legend(fontsize=7)
i = int(np.flatnonzero(sim == 12568)[0]); d = diag[i]; ok = np.isfinite(d[0])
ax[1].plot(idx, x5[i], "k", lw=1.5, label="slice-mean x$_{HI}$ (lightcone)")
ax[1].plot(idx, 1 - np.interp(zu, d[0][ok], d[1][ok]), "C2--", label="box x$_{HI}$ at uniform-z")
ax[1].plot(idx, 1 - np.interp(zc, d[0][ok], d[1][ok]), "C3--", label="box x$_{HI}$ at comoving z")
ax[1].set_xlim(0, 700); ax[1].set_xlabel("LOS plane index"); ax[1].set_ylabel("x$_{HI}$")
ax[1].set_title("(b) cone 12568 vs its 21ssd box history"); ax[1].legend(fontsize=7)
spacing_u = np.gradient(G["distance_native"][:]); spacing_c = np.gradient(dc)
ax[2].plot(idx, spacing_u, "C2", label="true (uniform z)"); ax[2].plot(idx, spacing_c, "C3", label="assumed (comoving)")
corr = []
for s in (12280, 19799, 13648):
    c = LoreliCone(W / f"data/loreli_lc128/lightcone_xa_tpHI_xion_dens_dtb_no_rsd_simu{s}.dat", l_perp=294.0)
    for a in (340, 560, 760, 1000, 1240, 1440):
        dl = c.delta(c.read(a, a + 64, fields=("dens",))["dens"]); dl = dl - dl.mean((0, 1), keepdims=True)
        corr.append((a + 32, np.mean(dl[:, :, 1:] * dl[:, :, :-1]) / np.mean(dl ** 2),
                     np.mean(dl[1:] * dl[:-1]) / np.mean(dl ** 2)))
    del c
corr = np.array(corr)
a2 = ax[2].twinx(); a2.grid(False)
a2.plot(corr[:, 0], corr[:, 1], "ko", ms=3, label="LOS lag-1 density corr.")
a2.axhline(np.mean(corr[:, 2]), color="grey", ls=":", label="transverse lag-1 (2.30 Mpc)")
a2.set_ylabel("lag-1 correlation"); a2.set_ylim(0.5, 0.8); a2.legend(fontsize=7, loc="lower right")
ax[2].set_xlabel("LOS plane index"); ax[2].set_ylabel("plane spacing [cMpc]"); ax[2].set_ylim(0, 4.5); ax[2].legend(fontsize=7, loc="upper right")
ax[2].set_title("(c) plane spacing and density correlation")
save(fig, "fig_los_geometry")
res = np.array([(k, np.median(est[k])) for k in range(1536) if len(est[k]) >= 5])
summary["z_est_rms"] = {n: float(np.sqrt(np.mean((res[:, 1] - g[res[:, 0].astype(int)]) ** 2)))
                        for n, g in (("uniform_z", zu), ("comoving", zc), ("frequency", zf))}
summary["lag1"] = corr.tolist()

# ---------------- cache rows: assumed vs true z, stages, splits
with h5py.File(CACHE, "r") as h:
    cz, split, stage, cone, los, xmean = (h[k][:] for k in ("z", "split", "stage", "cone_id", "los_index", "xhi_mean"))
    scal = h["scalars"][:]
    bands = json.loads(h.attrs["bands"])
true_z = np.interp(cz, zc, zu)
fig, ax = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
bins = np.linspace(5, 11, 61)
ax[0].hist(cz, bins, histtype="step", color="C3", label="as labelled in cache (comoving axis)")
ax[0].hist(true_z, bins, histtype="step", color="C2", label="true (uniform-z axis)")
ax[0].set_xlabel("z of cache row"); ax[0].set_ylabel("rows"); ax[0].legend(fontsize=7); ax[0].set_title("(a) redshift of the 77,182 rows")
names = ["outside stage bins\n(x$_{HI}$<0.05 or >0.95)", "x$_{HI}$ 0.05-0.35", "0.35-0.65", "0.65-0.95"]
for k, (s, lab) in enumerate(((0, "train"), (1, "val"), (2, "test"))):
    m = split == s
    counts = [((stage == -1) & m).sum()] + [((stage == j) & m).sum() for j in (0, 1, 2)]
    ax[1].bar(np.arange(4) + 0.27 * (k - 1), counts, 0.27, label=f"{lab} ({m.sum()} rows)")
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(names, fontsize=7); ax[1].set_ylabel("rows"); ax[1].legend(fontsize=7)
ax[1].set_title("(b) rows by stage and split")
save(fig, "fig_cache_rows")
summary["true_z_q"] = np.percentile(true_z, [1, 50, 99]).tolist()
summary["assumed_z_q"] = np.percentile(cz, [1, 50, 99]).tolist()

# ---------------- history leakage
k01 = bands.index([0, 1])
with h5py.File(SIDE, "r") as s:
    emu = s["scalars"][:]
FC = W / "data/compressed/rhizome_mf_slices_all.h5"
rng = np.random.default_rng(0)
with h5py.File(FC, "r") as f:
    fsplit = f["split"][:]
    fi = np.sort(rng.choice(np.flatnonzero(fsplit == 2), 1500, replace=False))
    fsc = f["scalars"][:][fi][:, 12 + k01]
    fxm = np.array([f["xhi"][j].astype(np.float32).mean() for j in fi])
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True, sharey=True)
act = (xmean > 0.02) & (xmean < 0.98)
for a, (x, y, t) in zip(ax, ((xmean, scal[:, 12 + k01], "LoReLi cache: oracle (cone's own) history"),
                             (xmean, emu[:, 12 + k01], "LoReLi side file: emulated history"),
                             (fxm, fsc, "21cmFAST training cache (test rows)"))):
    a.hexbin(x, y, gridsize=60, bins="log", cmap="viridis", mincnt=1, extent=(0, 1, 0, 1))
    a.plot([0, 1], [0, 1], "r-", lw=0.6)
    rms = np.sqrt(np.mean((x - y) ** 2))
    a.set_title(t, fontsize=8); a.set_xlabel("target slice mean x$_{HI}$")
    a.text(0.03, 0.9, f"RMS {rms:.3f}", transform=a.transAxes, color="w", fontsize=8)
ax[0].set_ylabel("history scalar x$_{HI}$, band (0,1)")
save(fig, "fig_history_leakage")
summary["leak_rms"] = {"oracle": float(np.sqrt(np.mean((xmean - scal[:, 12 + k01]) ** 2))),
                       "emulated": float(np.sqrt(np.mean((xmean - emu[:, 12 + k01]) ** 2))),
                       "21cmfast_test": float(np.sqrt(np.mean((fxm - fsc) ** 2)))}
summary["leak_rms_active"] = {"oracle": float(np.sqrt(np.mean((xmean - scal[:, 12 + k01])[act] ** 2))),
                              "emulated": float(np.sqrt(np.mean((xmean - emu[:, 12 + k01])[act] ** 2)))}

# ---------------- emulator examples and errors
P = h5py.File(R / "runs/loreli_emulator/v1/predictions.h5", "r")
zn = P["z_node"][:]; held = P["heldout"][:]
xb, tb = G["xhi_box"][:], G["tb_slice"][:]
ex, et = P["x_HI/full"][:], P["T_b/full"][:]
sx, st = P["x_HI/spread"][:], P["T_b/spread"][:]
cand = [j for j in np.flatnonzero(held) if np.isfinite(tb[j]).sum() > 120 and np.isfinite(xb[j]).sum() > 120]
pick = [cand[k] for k in np.linspace(0, len(cand) - 1, 4).astype(int)]
th = G["theta"][:]
fig, ax = plt.subplots(2, 4, figsize=(12, 5), constrained_layout=True, sharex=True)
for c, j in enumerate(pick):
    ax[0, c].plot(zn, xb[j], "k", lw=1.5, label="box (21ssd)")
    ax[0, c].plot(zn, ex[j], "C1--", label="emulator (full fit)")
    ax[0, c].fill_between(zn, ex[j] - 2 * sx[j], ex[j] + 2 * sx[j], color="C1", alpha=0.3, lw=0)
    if kind[j] == 2:
        ax[0, c].plot(zn, G["xhi_slice"][j], "C2:", label="slice mean (5-field)")
    ax[1, c].plot(zn, tb[j], "k", lw=1.5, label="lightcone slice mean")
    ax[1, c].plot(zn, et[j], "C1--", label="emulator")
    ax[1, c].fill_between(zn, et[j] - 2 * st[j], et[j] + 2 * st[j], color="C1", alpha=0.3, lw=0)
    t = th[j]
    ax[0, c].set_title(f"sim {sim[j]}: f$_X$={t[0]:g}, $\\tau$={t[1]:.0f}, r$_{{H/S}}$={t[2]:g},\nlogM={t[3]:g}, f$_{{esc}}$={t[4]:g}", fontsize=7)
    ax[1, c].set_xlabel("z")
ax[0, 0].set_ylabel("x$_{HI}$"); ax[1, 0].set_ylabel("T$_b$ [mK]")
ax[0, 0].legend(fontsize=6); ax[1, 0].legend(fontsize=6)
fig.suptitle("History emulator on held-out simulations (cache val/test blocks; never trained on), ±2σ ensemble spread", fontsize=9)
save(fig, "fig_emulator_examples")

r = json.load(open(R / "runs/loreli_emulator/v1/results.json"))["scores"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
for a, q, unit in ((ax[0], "x_HI", ""), (ax[1], "T_b", " [mK]")):
    zb = list(r[q]["heldout_full"]["z_band"])
    for k, (key, lab) in enumerate((("heldout_full", "held-out (full fit)"), ("pool_out_of_fold", "pool, out-of-fold"),
                                    ("pool_in_sample_full", "pool, in-sample"))):
        v = [r[q][key]["z_band"][b] or 0 for b in zb]
        a.bar(np.arange(len(zb)) + 0.27 * (k - 1), v, 0.27, label=lab)
    a.set_xticks(range(len(zb))); a.set_xticklabels([f"z {b}" for b in zb], fontsize=7)
    a.set_ylabel(f"RMS error{unit}"); a.set_title(f"{q} emulator error by redshift"); a.legend(fontsize=7)
save(fig, "fig_emulator_errors")

# ---------------- fine-tuning curves
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True, sharey=True)
for a, nb in ((ax[0], 44), (ax[1], 8)):
    for k, arm in enumerate(("decoder", "ends", "pointwise", "none", "scratch")):
        f = R / f"runs/loreli_finetune/l294_b{nb}_{arm}_seed0/metrics.jsonl"
        if not f.exists():
            continue
        m = [json.loads(l) for l in open(f)]
        m = [x for x in m if "val_score" in x]
        a.plot([x["step"] for x in m], [x["val_score"] for x in m], f"C{k}", lw=1, label=arm)
        summary.setdefault("ft", {})[f"b{nb}_{arm}"] = {"best": min(x["val_score"] for x in m), "last_step": m[-1]["step"],
                                                       "best_step": min(m, key=lambda x: x["val_score"])["step"],
                                                       "best_row": min(m, key=lambda x: x["val_score"])}
    a.axhline(0.0652, color="grey", ls=":", lw=1, label="before fine-tuning")
    a.set_yscale("log"); a.set_xlabel("step (batch 8)"); a.set_title(f"{nb} training blocks")
    a.legend(fontsize=7)
ax[0].set_ylabel("validation score (mean normalized MSE)")
save(fig, "fig_finetune_curves")

json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in summary.items() if k not in ("lag1",)}, indent=1, default=float)[:6000])
