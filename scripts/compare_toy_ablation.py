"""Summarize the toy Hyena/H3-comparison ablations of the tied spectral Rhizome.

Baseline: dense channel-mixing kernel, gates recomputed from the evolving
state (``--baseline``, seed 0) plus the ``dense_seed*`` runs in ``--sweep``,
which give the seed-to-seed spread. Ablations: per-channel ("depthwise")
kernel, gates frozen at h0, and a frozen message.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_toy_radius import cone_bootstrap_ratio, load

SHARED = ("steps", "batch_size", "modes", "updates", "step_size", "lr", "val_every", "threads")
DESCRIPTION = {
    "dense": "dense C x C kernel, recomputed gates (baseline)",
    "depthwise_w16": "per-channel kernel (Hyena-like), same width",
    "depthwise_w36": "per-channel kernel (Hyena-like), ~same parameters",
    "static_gates": "gates frozen at h0 = tanh(forcing)",
    "static_message": "gates and values frozen: one-shot message",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="seed-0 dense run directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs = {"dense_seed0": load(args.baseline)}
    for path in sorted(p for p in args.sweep.iterdir() if (p / "metadata.json").exists() and (p / "results.json").exists()):
        runs[path.name] = load(path)
    _, base_meta, base_result, base = runs["dense_seed0"]
    for name, (_, meta, _, fields) in runs.items():
        for key in SHARED:
            if meta["config"][key] != base_meta["config"][key]:
                raise ValueError(f"uncontrolled config {key}: {name}")
        if meta["cache"] != base_meta["cache"] or not np.array_equal(fields["truth"], base["truth"]):
            raise ValueError(f"different test data: {name}")

    dense = [runs[n][2]["test"]["all"]["rmse"] for n in sorted(runs) if n.startswith("dense_seed")]
    rows = []
    for name, (variant, meta, result, fields) in runs.items():
        group = "dense" if name.startswith("dense_seed") else name
        rows.append({
            "run": name, "group": group, "description": DESCRIPTION[group], "variant": variant,
            "width": meta["config"]["width"], "seed": meta["config"]["seed"],
            "parameters_real": result["parameters_real"], "best_step": result["best_step"],
            "best_validation_bce": result["best_validation_bce"], "test": result["test"],
            "rmse_ratio_vs_dense_seed0": result["test"]["all"]["rmse"] / base_result["test"]["all"]["rmse"],
            "rmse_ratio_cone_bootstrap_95": cone_bootstrap_ratio(
                fields["prediction"], base["prediction"], base["truth"], base["cone_id"]),
            "iteration_sweep": {k: v["all"]["rmse"] for k, v in result.get("iteration_sweep", {}).items()},
            "train_seconds_including_validation": result["train_seconds_including_validation"],
        })
    order = list(DESCRIPTION)
    rows.sort(key=lambda r: (order.index(r["group"]), r["seed"]))
    summary = {"comparability_checks_passed": True, "baseline": str(args.baseline),
               "dense_seed_rmse": {"values": dense, "mean": float(np.mean(dense)),
                                   "std": float(np.std(dense, ddof=1)) if len(dense) > 1 else None},
               "rows": rows,
               "interpretation": "Ablations use one seed; compare their effect with the dense seed spread. "
                                 "Bootstrap intervals cover test-cone sampling only."}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"dense RMSE over seeds {[round(v, 4) for v in dense]}: mean {np.mean(dense):.4f}"
          + (f", std {np.std(dense, ddof=1):.4f}" if len(dense) > 1 else ""))
    print(f"{'run':15s} {'w':>3s} {'params':>6s} {'valBCE':>7s} {'RMSE':>7s} {'RMSEmix':>7s} {'IoUmix':>6s} "
          f"{'x dense0':>8s} {'95% (cones)':>13s} {'step':>5s}  RMSE at T=1/2/4/8")
    for r in rows:
        lo, hi = r["rmse_ratio_cone_bootstrap_95"]
        sweep = "/".join(f"{r['iteration_sweep'][k]:.3f}" for k in ("1", "2", "4", "8") if k in r["iteration_sweep"])
        print(f"{r['run']:15s} {r['width']:3d} {r['parameters_real']:6d} {r['best_validation_bce']:7.4f} "
              f"{r['test']['all']['rmse']:7.4f} {r['test']['mixed']['rmse']:7.4f} {r['test']['mixed']['ionized_iou']:6.3f} "
              f"{r['rmse_ratio_vs_dense_seed0']:8.3f} {lo:6.3f}-{hi:.3f} {r['best_step']:5d}  {sweep}")


if __name__ == "__main__":
    main()
