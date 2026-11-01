"""Comparative figures from the 21cmEVAL runs on the LoReLi II v3 test lightcones.

    python docs/figures/loreli/src/make_loreli_cm21eval_figs.py [runs/cm21eval] [docs/figures/loreli/cm21eval]

Reads <root>/<model>/<group>/{xhi,tb}/{curves.npz,summary.csv} for every model present and
<root>/groups.json, and writes PDF + PNG figures.  The test cones are split into geometry
groups (one common z grid per group); everything here is pooled over all cones of all groups:
per-cone quantities are concatenated and summarized by the median (bands: 16-84 %), voxel
RMSE is pooled exactly (voxel-count weighted MSE), front profiles are count-weighted.
Models that are missing (e.g. 44-block arms still training) are skipped, so the script can be
rerun as results arrive.
"""
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

REPO = Path(__file__).resolve().parents[4]
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "runs/cm21eval"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "docs/figures/loreli/cm21eval"
OUT.mkdir(parents=True, exist_ok=True)

STYLE = {  # name: (label, color, linestyle)
    "reference": ("reference (zero-shot)", "0.15", "--"),
    "reference_oracle": ("reference, oracle hist.", "0.55", ":"),
    "b8_decoder": ("8 blk: decoder", "#9ecae1", "-"),
    "b8_ends": ("8 blk: ends", "#6baed6", "-"),
    "b8_pointwise": ("8 blk: pointwise", "#3182bd", "-"),
    "b8_none": ("8 blk: none", "#08519c", "-"),
    "b8_scratch": ("8 blk: scratch", "#08306b", "-."),
    "b44_decoder": ("44 blk: decoder", "#fcae91", "-"),
    "b44_ends": ("44 blk: ends", "#fb6a4a", "-"),
    "b44_pointwise": ("44 blk: pointwise", "#de2d26", "-"),
    "b44_none": ("44 blk: none", "#a50f15", "-"),
    "b44_scratch": ("44 blk: scratch", "#67000d", "-."),
    "b44_none_oracle": ("44 blk: none, oracle hist.", "#a50f15", ":"),
}
STAGES = ["0p02_0p20", "0p20_0p40", "0p40_0p60", "0p60_0p80", "0p80_0p98"]
STAGE_LABEL = {s: "$\\bar{x}_{HI}$ " + s.replace("p", ".").replace("_", "–") for s in STAGES}
SIGMA_TAU = 0.0073
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "legend.fontsize": 7, "figure.dpi": 100})

groups = json.loads((ROOT / "groups.json").read_text())
GROUPS = list(groups)
models = [m for m in STYLE if all((ROOT / m / g / "xhi" / "curves.npz").exists() for g in GROUPS)]
partial = [m for m in STYLE if m not in models and (ROOT / m).is_dir()]
if not models:
    raise SystemExit(f"no complete 21cmEVAL results under {ROOT}")
print("models:", models, "| incomplete (skipped):", partial)
lab = lambda m: STYLE[m][0]
tick = lambda m: lab(m).replace(", ", ",\n").replace(" (", "\n(").replace(": ", ":\n")
col = lambda m: STYLE[m][1]
ls = lambda m: STYLE[m][2]


def curves(m, g, target="xhi"):
    with np.load(ROOT / m / g / target / "curves.npz") as d:
        return {k.split("__")[-1]: d[k] for k in d.files}


def summary(m, g, target="xhi"):
    return next(csv.DictReader(open(ROOT / m / g / target / "summary.csv")))


def pooled(m, key, target="xhi"):
    """Concatenate a per-cone array over all groups (axis 0)."""
    parts = []
    for g in GROUPS:
        c = curves(m, g, target)
        if key in c:
            parts.append(np.atleast_1d(c[key]).astype(float))
    return np.concatenate(parts, 0) if parts else np.array([])


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)
    print("wrote", name)


def med_band(a, axis=0):
    a = np.asarray(a, float)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(a, axis), np.nanpercentile(a, 16, axis), np.nanpercentile(a, 84, axis)


def legend_handles(ax_or_fig, where, **kw):
    hs = [plt.Line2D([], [], color=col(m), ls=ls(m), lw=1.6, label=lab(m)) for m in models]
    return where.legend(handles=hs, **kw)


cone_group = [g for g in GROUPS for _ in groups[g]["cones"]]
n_cones = len(cone_group)
k = curves(models[0], GROUPS[0])["spectrum_k"]

# =============================================================================================
# 1) scorecard
# =============================================================================================
def pooled_rmse(m, target):
    n = mse = 0.0
    for g in GROUPS:
        s = summary(m, g, target)
        nv = float(s["voxel.n_voxels"])
        n += nv
        mse += float(s["voxel.mse"]) * nv
    return math.sqrt(mse / n)


def at_k(curve, k0=0.2):
    return np.array([np.interp(np.log(k0), np.log(k), row) if np.isfinite(row).any() else np.nan for row in curve])


metrics = []   # (title, per-model per-cone array or scalar, better: "low"/"high"/"one"/"zero", fmt)
for m in models:
    pass
score = {}
for m in models:
    tau_p, tau_t = pooled(m, "history_tau_pred"), pooled(m, "history_tau_truth")
    z50_p, z50_t = pooled(m, "history_z_xhi50_pred"), pooled(m, "history_z_xhi50_truth")
    fw_p, fw_t = pooled(m, "front_width_pred"), pooled(m, "front_width_truth")
    score[m] = {
        "x_HI RMSE (pooled)": pooled_rmse(m, "xhi"),
        "T_b RMSE [mK] (pooled)": pooled_rmse(m, "tb"),
        "x_HI RMSE per cone": pooled(m, "rmse_per_sample"),
        "T_b RMSE per cone [mK]": pooled(m, "rmse_per_sample", "tb"),
        "IoU (matched)": pooled(m, "morphology_iou_matched"),
        "x_HI r(k=0.2/Mpc), active": at_k(pooled(m, "spectrum_coherence_active_0p05_0p95")),
        "T_b r(k=0.2/Mpc), active": at_k(pooled(m, "spectrum_coherence_active_0p05_0p95", "tb")),
        "x_HI P ratio (k=0.2/Mpc)": at_k(pooled(m, "spectrum_ratio_active_0p05_0p95")),
        "|Δτ| [σ_Planck]": np.abs(tau_p - tau_t) / SIGMA_TAU,
        "Δz(x_HI=0.5)": z50_p - z50_t,
        "front width ratio": fw_p / fw_t,
    }
panels = [("x_HI RMSE (pooled)", "low"), ("T_b RMSE [mK] (pooled)", "low"), ("IoU (matched)", "high"),
          ("x_HI r(k=0.2/Mpc), active", "high"), ("T_b r(k=0.2/Mpc), active", "high"),
          ("x_HI P ratio (k=0.2/Mpc)", "one"), ("|Δτ| [σ_Planck]", "low"), ("Δz(x_HI=0.5)", "zero"),
          ("front width ratio", "one")]
fig, axs = plt.subplots(3, 3, figsize=(13, 9.5), constrained_layout=True)
y = np.arange(len(models))[::-1]
for ax, (name, better) in zip(axs.ravel(), panels):
    for i, m in enumerate(models):
        v = score[m][name]
        if np.ndim(v) == 0:
            ax.barh(y[i], v, color=col(m), edgecolor="k", lw=0.4)
            ax.text(v, y[i], f" {v:.3g}", va="center", fontsize=7)
        else:
            md, lo, hi = med_band(v)
            ax.barh(y[i], md, color=col(m), edgecolor="k", lw=0.4)
            ax.errorbar(md, y[i], xerr=[[md - lo], [hi - md]], color="k", lw=0.8, capsize=2)
            jitter = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.5
            ax.scatter(v, y[i] + jitter, s=4, color="k", alpha=0.35, lw=0, zorder=3)
            ax.text(hi, y[i] + 0.3, f" {md:.3g}", va="center", fontsize=7)
    if better == "one":
        ax.axvline(1, color="k", lw=0.6)
    if better == "zero":
        ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([lab(m) for m in models] if ax in axs[:, 0] else [])
    hint = {"low": "lower is better", "high": "higher is better", "one": "1 is ideal", "zero": "0 is ideal"}[better]
    ax.set_title(f"{name}  ({hint})")
fig.suptitle(f"21cmEVAL scorecard, LoReLi II v3 test set: {n_cones} lightcones in {len(GROUPS)} redshift groups, "
             "emulated histories unless 'oracle'. Bars: pooled value or median over cones (error bar 16–84 %, dots: cones)",
             fontsize=9.5)
save(fig, "cm21_scorecard")

# =============================================================================================
# 2) per-cone RMSE, paired across models
# =============================================================================================
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
gcol = dict(zip(GROUPS, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]))
for ax, key, unit in ((axs[0], "x_HI RMSE per cone", ""), (axs[1], "T_b RMSE per cone [mK]", " [mK]")):
    vals = np.array([score[m][key] for m in models])            # (models, cones)
    xs = np.arange(len(models))
    for c in range(vals.shape[1]):
        ax.plot(xs, vals[:, c], "-", color=gcol[cone_group[c]], alpha=0.45, lw=0.8)
    ax.boxplot(vals.T, positions=xs, widths=0.35, showfliers=False, medianprops={"color": "k"})
    ax.set_xticks(xs)
    ax.set_xticklabels([tick(m) for m in models], fontsize=7)
    ax.set_title(f"{key.replace(' per cone', '')} per lightcone{''}: lines connect the same cone")
    ax.set_ylabel(key)
hs = [plt.Line2D([], [], color=gcol[g], label=f"{g} ({len(groups[g]['cones'])} cones)") for g in GROUPS]
axs[1].legend(handles=hs, loc="upper right")
save(fig, "cm21_per_cone_rmse")

# =============================================================================================
# 3/4) power-spectrum ratio and cross-correlation per stage (x_HI and T_b)
# =============================================================================================
for target, tname in (("xhi", "x$_{HI}$"), ("tb", "T$_b$")):
    fig, axs = plt.subplots(2, 6, figsize=(17, 6), constrained_layout=True, sharex=True)
    for j, st in enumerate(STAGES + ["active"]):
        key = "active_0p05_0p95" if st == "active" else f"xbar_HI_{st}"
        for m in models:
            r = pooled(m, f"spectrum_ratio_{key}", target)
            c = pooled(m, f"spectrum_coherence_{key}", target)
            for row, a in ((0, r), (1, c)):
                md, lo, hi = med_band(a)
                axs[row, j].plot(k, md, color=col(m), ls=ls(m), lw=1.4)
                if m in ("reference", models[-1]):
                    axs[row, j].fill_between(k, lo, hi, color=col(m), alpha=0.12, lw=0)
        n = np.isfinite(pooled(models[0], f"spectrum_ratio_{key}", target)).any(1).sum()
        axs[0, j].set_title(("active $\\bar{x}_{HI}$ 0.05–0.95" if st == "active" else STAGE_LABEL[st]) + f"  ({n} cones)")
        axs[0, j].axhline(1, color="k", lw=0.6)
        axs[1, j].axhline(1, color="k", lw=0.6)
        axs[0, j].set_yscale("log")
        axs[0, j].set_ylim(0.1, 10)
        axs[1, j].set_ylim(-0.1, 1.02)
        axs[1, j].set_xscale("log")
        axs[1, j].set_xlabel("k$_\\perp$ [Mpc$^{-1}$]")
    axs[0, 0].set_ylabel("P$_{pred}$ / P$_{true}$")
    axs[1, 0].set_ylabel("cross-correlation r(k)")
    legend_handles(fig, axs[1, 0], loc="lower left")
    fig.suptitle(f"{tname}: transverse power ratio and coherence by reionization stage; median over cones "
                 f"(band 16–84 % for the reference and {lab(models[-1])})", fontsize=10)
    save(fig, f"cm21_spectra_{target}")

# =============================================================================================
# 5) bubble-size distributions per stage
# =============================================================================================
R = curves(models[0], GROUPS[0])["bubble_radius_mpc"]
fig, axs = plt.subplots(1, 6, figsize=(17, 3.6), constrained_layout=True, sharey=True)
for j, st in enumerate(STAGES + ["active"]):
    key = "active" if st == "active" else f"xbar{st.replace('p', '.').replace('_', '_')}"
    key = "active" if st == "active" else "xbar" + st.replace("p", ".")
    truth = np.nanmean(pooled(models[0], f"bubble_mass_truth_{key}"), 0)
    axs[j].plot(R, truth, color="k", lw=2.2, label="truth")
    for m in models:
        axs[j].plot(R, np.nanmean(pooled(m, f"bubble_mass_pred_{key}"), 0), color=col(m), ls=ls(m), lw=1.3)
    axs[j].set_xscale("log")
    axs[j].set_xlabel("mean-free-path size R [Mpc]")
    axs[j].set_title("active 0.05–0.95" if st == "active" else STAGE_LABEL[st])
axs[0].set_ylabel("ionized-area-weighted PDF (per bin)")
hs = [plt.Line2D([], [], color="k", lw=2.2, label="truth")] + \
     [plt.Line2D([], [], color=col(m), ls=ls(m), label=lab(m)) for m in models]
axs[0].legend(handles=hs, loc="upper left")
fig.suptitle("Bubble-size distributions (mean-free-path rays through transverse slices), mean over cones", fontsize=10)
save(fig, "cm21_bubbles")

# =============================================================================================
# 6) reionization history: tau, midpoint, duration, example histories
# =============================================================================================
fig = plt.figure(figsize=(16, 8.2), constrained_layout=True)
gs = fig.add_gridspec(2, 4)
ax_tau, ax_z50, ax_dz = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
ax_err = fig.add_subplot(gs[0, 3])
for m in models:
    for ax, kp, kt, sc in ((ax_tau, "history_tau_pred", "history_tau_truth", 1.0),
                           (ax_z50, "history_z_xhi50_pred", "history_z_xhi50_truth", 1.0),
                           (ax_dz, "history_delta_z_pred", "history_delta_z_truth", 1.0)):
        ax.scatter(pooled(m, kt), pooled(m, kp), s=10, color=col(m), alpha=0.8, lw=0, label=lab(m))
for ax, name in ((ax_tau, "τ$_e$ over the group window"), (ax_z50, "z(x$_{HI}$ = 0.5)"),
                 (ax_dz, "duration Δz = z(0.75) − z(0.25)")):
    lo, hi = ax.get_xlim()
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.6)
    ax.set_xlabel(f"truth: {name}")
    ax.set_ylabel("prediction")
    ax.set_title(name)
# error summary per model on the three quantities
qs = [("history_tau_pred", "history_tau_truth", SIGMA_TAU, "Δτ / σ$_{Planck}$"),
      ("history_z_xhi25_pred", "history_z_xhi25_truth", 1, "Δz(0.25)"),
      ("history_z_xhi50_pred", "history_z_xhi50_truth", 1, "Δz(0.5)"),
      ("history_z_xhi75_pred", "history_z_xhi75_truth", 1, "Δz(0.75)")]
w = 0.8 / len(models)
for i, m in enumerate(models):
    for q, (kp, kt, s, _) in enumerate(qs):
        e = (pooled(m, kp) - pooled(m, kt)) / s
        md, lo, hi = med_band(e)
        ax_err.errorbar(q + (i - len(models) / 2) * w, md, yerr=[[md - lo], [hi - md]], fmt="o", ms=3.5,
                        color=col(m), capsize=1.5, lw=0.9)
ax_err.axhline(0, color="k", lw=0.6)
ax_err.set_xticks(range(len(qs)))
ax_err.set_xticklabels([q[3] for q in qs])
ax_err.set_title("history errors: median and 16–84 % over cones")
# example histories, one cone per group
for gi, g in enumerate(GROUPS):
    ax = fig.add_subplot(gs[1, gi])
    c0 = curves(models[0], g)
    z = c0["history_z"]
    i0 = int(np.argmax(np.nanstd(c0["observable_xbar_truth"], 1)))       # the cone with the most evolution
    ax.plot(z, c0["observable_xbar_truth"][i0], color="k", lw=2.2, label="truth")
    for m in models:
        ax.plot(z, curves(m, g)["observable_xbar_pred"][i0], color=col(m), ls=ls(m), lw=1.2)
    ax.set_xlabel("z")
    ax.set_title(f"{g}: cone {groups[g]['cones'][i0]}")
    ax.set_ylim(-0.02, 1.02)
    if gi == 0:
        ax.set_ylabel("slice-mean x$_{HI}$")
legend_handles(fig, ax_tau, loc="upper left")
fig.suptitle("Reionization history recovered from the predicted maps (21cmEVAL `history` / `observable`). "
             "τ and crossings are computed inside each group's z window, so they compare prediction and truth, "
             "not the full reionization history", fontsize=9.5)
save(fig, "cm21_history")

# =============================================================================================
# 7) ionization fronts
# =============================================================================================
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)


def front_pool(m, key):
    num = den = 0
    for g in GROUPS:
        c = curves(m, g)
        n = c["front_counts"]
        v = c[key] ** 2 if key == "front_rmse" else c[key]
        num = num + n * v
        den = den + n
    out = num / np.maximum(den, 1)
    return np.sqrt(out) if key == "front_rmse" else out


d = curves(models[0], GROUPS[0])["front_d_mpc"]
axs[0].plot(d, front_pool(models[0], "front_mean_truth"), "k", lw=2.2, label="truth")
for m in models:
    axs[0].plot(d, front_pool(m, "front_mean_pred"), color=col(m), ls=ls(m), lw=1.3)
    axs[1].plot(d, front_pool(m, "front_rmse"), color=col(m), ls=ls(m), lw=1.3)
    axs[2].scatter(pooled(m, "front_width_truth"), pooled(m, "front_width_pred"), s=10, color=col(m), lw=0, alpha=0.85)
axs[0].set_title("mean x$_{HI}$ vs signed distance to the true front")
axs[0].set_xlabel("distance to true front [Mpc] (< 0: ionized side)")
axs[1].set_title("RMSE vs distance to the true front")
axs[1].set_xlabel("distance to true front [Mpc]")
axs[2].set_title("10–90 % front width per cone")
axs[2].set_xlabel("truth [Mpc]")
axs[2].set_ylabel("prediction [Mpc]")
lo, hi = axs[2].get_xlim()
axs[2].plot([lo, hi], [lo, hi], "k-", lw=0.6)
axs[2].text(hi, hi, "y = x ", ha="right", va="bottom", fontsize=7)
axs[2].set_xlim(lo, hi)
legend_handles(fig, axs[1], loc="upper right")
save(fig, "cm21_fronts")

# =============================================================================================
# 8) EFT bias / stochasticity and morphology
# =============================================================================================
fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
ek_all = curves(models[0], GROUPS[0])["eft_k"]
keep = ek_all <= 0.5              # the truth transfer function crosses zero near k ~ 1: ratios blow up there
ek = ek_all[keep]
for m in models:
    Tp, Tt = pooled(m, "eft_transfer_pred_active_0p05_0p95"), pooled(m, "eft_transfer_truth_active_0p05_0p95")
    Ep, Et = pooled(m, "eft_eps_pred_active_0p05_0p95"), pooled(m, "eft_eps_truth_active_0p05_0p95")
    rp, rt = pooled(m, "eft_r_pred_active_0p05_0p95"), pooled(m, "eft_r_truth_active_0p05_0p95")
    for ax, a in ((axs[0, 0], Tp / Tt), (axs[0, 1], Ep / Et), (axs[0, 2], rp - rt)):
        ax.plot(ek, med_band(a)[0][keep], color=col(m), ls=ls(m), lw=1.3)
for ax, t, ref in ((axs[0, 0], "transfer T(k) = P$_{x\\delta}$/P$_{\\delta\\delta}$: pred / truth", 1),
                   (axs[0, 1], "stochasticity P$_\\epsilon$(k): pred / truth", 1),
                   (axs[0, 2], "r$_{x\\delta}$(k): pred − truth", 0)):
    ax.axhline(ref, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("k [Mpc$^{-1}$]")
    ax.set_title(t + "\n(active slices, median over cones, k ≤ 0.5 Mpc$^{-1}$)")
axs[0, 1].set_yscale("log")
fs = [0.1, 0.25, 0.5, 0.75, 0.9]
for m in models:
    md = [med_band(pooled(m, f"morphology_iou_f{f}"))[0] for f in fs]
    axs[1, 0].plot(fs, md, marker="o", color=col(m), ls=ls(m), ms=3.5, lw=1.2)
    cnt = pooled(m, "morphology_pred_count") / pooled(m, "morphology_truth_count")
    rad = pooled(m, "morphology_pred_volume_weighted_radius_cells") / pooled(m, "morphology_truth_volume_weighted_radius_cells")
    for ax, a in ((axs[1, 1], cnt), (axs[1, 2], rad)):
        mdn, lo, hi = med_band(a)
        i = models.index(m)
        ax.errorbar(i, mdn, yerr=[[mdn - lo], [hi - mdn]], fmt="o", color=col(m), capsize=2)
axs[1, 0].set_xlabel("ionized volume fraction of the cone (threshold rank)")
axs[1, 0].set_title("3-D ionized-region IoU at matched ionized fraction")
for ax, t in ((axs[1, 1], "number of ionized regions: pred / truth"),
              (axs[1, 2], "volume-weighted region radius: pred / truth")):
    ax.axhline(1, color="k", lw=0.6)
    ax.set_yscale("log")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([tick(m) for m in models], fontsize=6.5)
    ax.set_title(t + " (median, 16–84 %)")
legend_handles(fig, axs[0, 0], loc="lower left")
save(fig, "cm21_eft_morphology")

# numbers behind the scorecard
tab = {m: {kk: (float(v) if np.ndim(v) == 0 else {"median": float(med_band(v)[0]), "p16": float(med_band(v)[1]),
                                                  "p84": float(med_band(v)[2]), "n": int(np.isfinite(v).sum())})
           for kk, v in score[m].items()} for m in models}
(OUT / "cm21_scorecard.json").write_text(json.dumps({"models": models, "skipped_incomplete": partial,
                                                     "groups": {g: groups[g]["cones"] for g in GROUPS},
                                                     "scores": tab}, indent=1, ensure_ascii=False))
print("wrote cm21_scorecard.json")
