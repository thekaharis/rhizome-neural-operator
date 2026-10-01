"""Summarize the toy x_HI sweep of compact (radius-R) versus spectral Rhizome kernels.

Each run directory under ``--sweep`` is a ``train_recurrent`` run with one
variant (``rhizome`` = spectral, ``rhizome_r<R>`` = compact kernel) and
otherwise identical configuration. Metrics are recomputed from the saved test
predictions. Uncertainty is a bootstrap over test *cones*, since slices from
one cone are correlated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REFERENCE = "rhizome"
SHARED = ("steps", "batch_size", "width", "modes", "updates", "step_size", "lr", "val_every", "seed", "threads")


def load(path):
    metadata = json.loads((path / "metadata.json").read_text())
    (variant,) = metadata["variants"]
    result = json.loads((path / "results.json").read_text())[variant]
    with np.load(path / variant / "test_predictions.npz") as arrays:
        fields = {k: arrays[k].copy() for k in ("prediction", "truth", "row", "cone_id")}
    rmse = float(np.sqrt(np.mean((fields["prediction"] - fields["truth"]) ** 2)))
    if not np.isclose(rmse, result["test"]["all"]["rmse"], rtol=1e-6):
        raise ValueError(f"saved metrics disagree for {path}")
    return variant, metadata, result, fields


def cone_bootstrap_ratio(pred, ref, truth, cones, n=10000, seed=0):
    """95% interval of RMSE(pred)/RMSE(ref), resampling whole test cones."""
    ids = np.unique(cones)
    sq = {c: ((pred[cones == c] - truth[cones == c]) ** 2).sum() for c in ids}
    sq_ref = {c: ((ref[cones == c] - truth[cones == c]) ** 2).sum() for c in ids}
    draws = np.random.default_rng(seed).choice(ids, (n, len(ids)))
    ratio = np.sqrt(np.array([sum(sq[c] for c in d) / sum(sq_ref[c] for c in d) for d in draws]))
    return [float(q) for q in np.quantile(ratio, [0.025, 0.975])]


def summarize(sweep, output):
    runs = {}
    for path in sorted(p for p in sweep.iterdir() if (p / "metadata.json").exists() and (p / "results.json").exists()):
        variant, *rest = load(path)
        runs[variant] = rest
    if REFERENCE not in runs:
        raise ValueError("missing spectral reference run 'rhizome'")
    base_meta, _, base = runs[REFERENCE]
    rows = []
    for variant, (meta, result, fields) in runs.items():
        for key in SHARED:
            if meta["config"][key] != base_meta["config"][key]:
                raise ValueError(f"uncontrolled config {key}: {variant}")
        if meta["cache"] != base_meta["cache"] or meta["splits"] != base_meta["splits"]:
            raise ValueError(f"different data: {variant}")
        for key in ("row", "truth"):
            if not np.array_equal(fields[key], base[key]):
                raise ValueError(f"test data/ordering differs: {variant}")
        radius = None if variant == REFERENCE else float(variant.removeprefix("rhizome_r"))
        per_slice = np.sqrt(((fields["prediction"] - fields["truth"]) ** 2).mean((1, 2)))
        per_slice_ref = np.sqrt(((base["prediction"] - base["truth"]) ** 2).mean((1, 2)))
        rows.append({
            "run": variant, "radius": radius,
            "reach_cells": None if radius is None else radius * meta["config"]["updates"],
            "parameters_real": result["parameters_real"], "best_step": result["best_step"],
            "best_validation_bce": result["best_validation_bce"], "test": result["test"],
            "train_seconds_including_validation": result["train_seconds_including_validation"],
            "vs_spectral": {
                "rmse_ratio": result["test"]["all"]["rmse"] / runs[REFERENCE][1]["test"]["all"]["rmse"],
                "rmse_ratio_cone_bootstrap_95": cone_bootstrap_ratio(
                    fields["prediction"], base["prediction"], base["truth"], base["cone_id"]),
                "slices_with_lower_rmse": int(np.sum(per_slice < per_slice_ref)),
                "slices_with_higher_rmse": int(np.sum(per_slice > per_slice_ref))},
        })
    rows.sort(key=lambda r: (r["radius"] is None, r["radius"] or 0))
    summary = {"comparability_checks_passed": True, "reference": REFERENCE,
               "shared_config": {k: base_meta["config"][k] for k in SHARED}, "cache": base_meta["cache"],
               "test_cones": base_meta["splits"]["test"]["cones"], "runs": rows,
               "interpretation": "One training seed per kernel; intervals cover test-cone sampling only. "
                                 "Timings come from runs executed in parallel and are observational."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    figure(rows, output / "radius_vs_spectral.png")
    print(f"{'run':14s} {'R':>5s} {'reach':>5s} {'params':>6s} {'valBCE':>7s} {'RMSE':>7s} {'RMSEmix':>7s} "
          f"{'IoUmix':>6s} {'x spec':>6s} {'95% (cones)':>13s} {'better':>6s} {'step':>5s}")
    for r in rows:
        v = r["vs_spectral"]
        print(f"{r['run']:14s} {r['radius'] or 0:5.2f} {r['reach_cells'] or 0:5.1f} {r['parameters_real']:6d} "
              f"{r['best_validation_bce']:7.4f} {r['test']['all']['rmse']:7.4f} {r['test']['mixed']['rmse']:7.4f} "
              f"{r['test']['mixed']['ionized_iou']:6.3f} {v['rmse_ratio']:6.3f} "
              f"{v['rmse_ratio_cone_bootstrap_95'][0]:6.3f}-{v['rmse_ratio_cone_bootstrap_95'][1]:.3f} "
              f"{v['slices_with_lower_rmse']:6d} {r['best_step']:5d}")
    return summary


def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    local = [r for r in rows if r["radius"] is not None]
    spectral = next(r for r in rows if r["radius"] is None)
    radius = [r["radius"] for r in local]
    params = [r["parameters_real"] for r in local]
    error = [r["test"]["all"]["rmse"] for r in local]
    spec_error = spectral["test"]["all"]["rmse"]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    ax = axes[0]
    ax.axhline(spec_error, color=orange, lw=2, ls="--",
               label=f"spectral, 4 modes ({spectral['parameters_real']:,} params)")
    ax.plot(radius, error, color=blue, lw=2, marker="o", ms=8, mec="white", mew=2, label="compact disk kernel")
    ax.set_xlabel("kernel radius R [cells]  (reach after 4 updates = 4R; grid 32, periodic)")
    ax.set_ylabel("test x_HI RMSE, all slices")
    ax.set_title("Error vs kernel radius", color=ink, loc="left")
    ax.legend(frameon=False, labelcolor=ink)
    ax = axes[1]
    ax.plot(params, error, color=blue, lw=2, marker="o", ms=8, mec="white", mew=2)
    for r, p, e in zip(radius, params, error):
        near = abs(np.log(p / spectral["parameters_real"])) < 0.25
        ax.annotate(f"R={r:.3g}", (p, e), textcoords="offset points", xytext=(-14, -16) if near else (6, 6),
                    color=muted, fontsize=9)
    ax.plot([spectral["parameters_real"]], [spec_error], ls="none", marker="D", ms=9, color=orange,
            mec="white", mew=2)
    ax.annotate("spectral", (spectral["parameters_real"], spec_error), textcoords="offset points",
                xytext=(4, 9), color=muted, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("trainable real parameters (whole model)")
    ax.set_title("Error vs parameter count", color=ink, loc="left")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.sweep, args.out)


if __name__ == "__main__":
    main()
