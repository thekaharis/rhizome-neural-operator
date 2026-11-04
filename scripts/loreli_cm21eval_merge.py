"""Merge per-model, per-group 21cmEVAL results into one comparison.

    python scripts/loreli_cm21eval_merge.py runs/cm21eval

Input:  runs/cm21eval/<model>/<group>/{xhi,tb}/{summary,stages}.csv  (one model per directory)
Output: compare_xhi.csv / compare_tb.csv  all summary columns, one row per (model, group)
        compare_stages.csv                long-form stage tables, both targets
        compare.md                        pooled voxel RMSE / bias over all groups (voxel-count weighted),
                                          headline metrics per group, provenance
Medians (spectrum, bubble, history...) cannot be pooled across groups and are reported per group.
"""
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
ORDER = ["reference", "reference_oracle", "b8_decoder", "b8_ends", "b8_pointwise", "b8_none", "b8_scratch",
         "b44_decoder", "b44_ends", "b44_pointwise", "b44_none", "b44_scratch", "b44_none_oracle"]
HEAD = {"xhi": [("voxel.rmse", "RMSE"), ("voxel.r2", "R²"), ("morphology.iou.matched", "IoU (matched)"),
                ("spectrum.active.coherence_ref", "r(k≈0.2)"), ("voxel.high_k_power_ratio", "high-k P ratio"),
                ("bubble.active.wasserstein_med_mpc", "bubble W1 [Mpc]"), ("bubble.active.relative_bias_med", "bubble size bias"),
                ("history.tau.error_sigma_med", "Δτ [σ_Planck]"), ("history.history.z_xhi50.error_med", "Δz(x=0.5)"),
                ("front.front_width_truth_mpc", "front width truth [Mpc]"), ("front.front_width_pred_mpc", "front width pred [Mpc]")],
        "tb": [("voxel.rmse", "RMSE [mK]"), ("voxel.r2", "R²"), ("voxel.bias", "bias [mK]"),
               ("spectrum.active.coherence_ref", "r(k≈0.2)"), ("voxel.high_k_power_ratio", "high-k P ratio"),
               ("edge.global_history_rmse", "global T_b RMSE [mK]")]}
LABEL = {"xhi": "x_HI", "tb": "T_b [mK]"}


def fmt(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—" if v in (None, "") else str(v)
    return "—" if math.isnan(x) else f"{x:.4g}"


models = sorted((p.name for p in root.iterdir() if p.is_dir() and any(p.glob("*/xhi/summary.csv"))),
                key=lambda m: (ORDER.index(m) if m in ORDER else 99, m))
groups = sorted({q.parent.parent.name for m in models for q in (root / m).glob("*/xhi/summary.csv")})
md = ["# 21cmEVAL: rhizome on LoReLi II v3 test lightcones", "",
      f"Models: {len(models)}; geometry groups: {', '.join(groups)}.", ""]
stages_out = []
for target in ("xhi", "tb"):
    rows = []
    for m in models:
        for gname in groups:
            f = root / m / gname / target / "summary.csv"
            if not f.exists():
                continue
            for r in csv.DictReader(open(f)):
                rows.append({"model": m, "group": gname, **{k: v for k, v in r.items() if k != "model"}})
            sf = root / m / gname / target / "stages.csv"
            if sf.exists():
                for r in csv.DictReader(open(sf)):
                    stages_out.append({"target": target, "model": m, "group": gname,
                                       **{k: v for k, v in r.items() if k != "model"}})
    if not rows:
        continue
    cols = ["model", "group"] + sorted({k for r in rows for k in r if k not in ("model", "group")})
    with open(root / f"compare_{target}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    # pooled voxel statistics (exact: MSE and bias weighted by voxel count)
    md += [f"## {LABEL[target]}: pooled over all groups", "",
           "| model | groups | RMSE | bias | voxels |", "|---|---|---|---|---|"]
    for m in models:
        rs = [r for r in rows if r["model"] == m and r.get("voxel.n_voxels")]
        n = sum(float(r["voxel.n_voxels"]) for r in rs)
        if not n:
            continue
        mse = sum(float(r["voxel.mse"]) * float(r["voxel.n_voxels"]) for r in rs) / n
        bias = sum(float(r["voxel.bias"]) * float(r["voxel.n_voxels"]) for r in rs) / n
        md.append(f"| {m} | {len(rs)}/{len(groups)} | {math.sqrt(mse):.4g} | {bias:+.3g} | {n:,.0f} |")
    md.append("")
    head = [(k, lab) for k, lab in HEAD[target] if any(k in r for r in rows)]
    for gname in groups:
        grows = [r for r in rows if r["group"] == gname]
        if not grows:
            continue
        md += [f"### {LABEL[target]}, group {gname}", "",
               "| model | " + " | ".join(lab for _, lab in head) + " |", "|---" * (len(head) + 1) + "|"]
        md += ["| " + r["model"] + " | " + " | ".join(fmt(r.get(k)) for k, _ in head) + " |" for r in grows]
        md.append("")
if stages_out:
    cols = ["target", "model", "group"] + sorted({k for r in stages_out for k in r} - {"target", "model", "group"})
    with open(root / "compare_stages.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(stages_out)
prov = {m: json.loads((root / "preds" / m / "provenance.json").read_text())
        for m in models if (root / "preds" / m / "provenance.json").exists()}
if prov:
    md += ["## Provenance", "", "| model | run | best step | histories | forward ms/plane |", "|---|---|---|---|---|"]
    md += [f"| {m} | `{Path(p['run']).name}` | {p['step']} | {p['histories'].split(' (')[0]} | {p['ms_per_plane']:.1f} |"
           for m, p in prov.items()]
    any_p = next(iter(prov.values()))
    md += ["", "Windows: " + "; ".join(f"{k}: {v['n']} planes, z {v['z'][0]:.2f}-{v['z'][1]:.2f}"
                                       for k, v in any_p["windows"].items()),
           f"Exactness vs cache rows (max |diff|): {any_p['check']}"]
(root / "compare.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
