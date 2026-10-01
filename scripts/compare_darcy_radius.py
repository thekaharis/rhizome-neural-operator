"""Summarize the Darcy sweep of compact (radius-R) versus spectral Rhizome kernels.

Each run directory under ``--sweep`` is a completed ``train_darcy`` run that
differs from the spectral run only in ``radius``. Metrics are recomputed from
saved predictions and checked against results.json; comparisons are paired
per test field against the spectral run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_darcy_widths import SHARED_CONFIG, load_run


def summarize(sweep, reference, output):
    paths = sorted(p for p in sweep.iterdir() if (p / "metadata.json").exists() and (p / "results.json").exists())
    runs = {p.name: load_run(p) for p in paths}
    if reference not in runs:
        raise ValueError(f"missing reference run {reference!r}")
    base_meta, base_result, base_saved, base_error = runs[reference]
    if base_meta["config"].get("radius", 0.0):
        raise ValueError("reference run must use the spectral kernel")
    rows = []
    for name, (meta, result, saved, error) in runs.items():
        for key in ("files_sha256", "train_indices", "val_indices", "stats"):
            if meta[key] != base_meta[key]:
                raise ValueError(f"uncontrolled {key}: {name}")
        for key in SHARED_CONFIG + ("width",):
            if meta["config"][key] != base_meta["config"][key]:
                raise ValueError(f"uncontrolled config {key}: {name}")
        for key in ("row", "truth", "coefficient"):
            if not np.array_equal(saved[key], base_saved[key]):
                raise ValueError(f"test data/ordering differs for {name}: {key}")
        radius = meta["config"].get("radius", 0.0) or None
        rows.append({
            "run": name, "radius": radius,
            "reach_cells": None if radius is None else radius * meta["config"]["updates"],
            "parameters_real": result["parameters_real"], "best_step": result["best_step"],
            "best_validation_relative_l2": result["best_validation_relative_l2"],
            "test": result["test"],
            "train_seconds_including_validation": result["train_seconds_including_validation"],
            "vs_spectral": {
                "mean_relative_l2_ratio": float(error.mean() / base_error.mean()),
                "fields_with_lower_error": int(np.sum(error < base_error)),
                "fields_with_higher_error": int(np.sum(error > base_error)),
                "parameter_ratio": result["parameters_real"] / base_result["parameters_real"]},
        })
    rows.sort(key=lambda r: (r["radius"] is None, r["radius"] or 0))
    summary = {
        "comparability_checks_passed": True, "reference": reference,
        "shared_config": {k: base_meta["config"][k] for k in SHARED_CONFIG + ("width",)},
        "runs": rows,
        "interpretation": "One seed per kernel. Timings come from runs executed four at a time on one machine "
                          "and are observational. The test set was inspected in earlier work.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output / "per_field_relative_l2.npz", row=base_saved["row"],
                        **{name: values[3] for name, values in runs.items()})
    figure(rows, output / "radius_vs_spectral.png")
    print(f"{'run':10s} {'R':>5s} {'reach':>5s} {'params':>7s} {'val':>7s} {'test':>7s} {'p95':>7s} "
          f"{'x spec':>6s} {'better':>6s} {'min':>5s}")
    for r in rows:
        print(f"{r['run']:10s} {r['radius'] or 0:5.2f} {r['reach_cells'] or 0:5.1f} {r['parameters_real']:7d} "
              f"{100 * r['best_validation_relative_l2']:6.3f}% {100 * r['test']['relative_l2_mean']:6.3f}% "
              f"{100 * r['test']['relative_l2_p95']:6.3f}% {r['vs_spectral']['mean_relative_l2_ratio']:6.3f} "
              f"{r['vs_spectral']['fields_with_lower_error']:6d} "
              f"{r['train_seconds_including_validation'] / 60:5.1f}")
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
    error = [100 * r["test"]["relative_l2_mean"] for r in local]
    spec_error = 100 * spectral["test"]["relative_l2_mean"]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    ax = axes[0]
    ax.axhline(spec_error, color=orange, lw=2, ls="--", label=f"spectral, 8 modes ({spectral['parameters_real']:,} params)")
    ax.plot(radius, error, color=blue, lw=2, marker="o", ms=8, mec="white", mew=2, label="compact disk kernel")
    ax.set_xlabel("kernel radius R [cells]  (reach after 4 updates = 4R; grid 32)")
    ax.set_ylabel("mean test relative L2 [%]")
    ax.set_title("Error vs kernel radius", color=ink, loc="left")
    ax.legend(frameon=False, labelcolor=ink)
    ax = axes[1]
    ax.plot(params, error, color=blue, lw=2, marker="o", ms=8, mec="white", mew=2)
    for r, p, e in zip(radius, params, error):
        # Points near the spectral marker are labelled below-left to avoid a collision.
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
    parser.add_argument("--reference", default="spectral")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.sweep, args.reference, args.out)


if __name__ == "__main__":
    main()
