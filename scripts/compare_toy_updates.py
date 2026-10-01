"""Compare the toy R=2 update-count sweep with the T=4 radius sweep at equal reach.

Reach after T updates of a radius-R kernel is R*T cells. ``--updates-sweep``
holds R=2 runs at several T (directories ``T<T>``); ``--radius-sweep`` holds
the T=4 runs from ``compare_toy_radius`` (including R=2, T=4 and spectral).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_toy_radius import cone_bootstrap_ratio, load

SHARED = ("steps", "batch_size", "width", "modes", "step_size", "lr", "val_every", "seed", "threads")


def collect(updates_sweep, radius_sweep):
    runs = []
    for series, sweep in (("vary_T", updates_sweep), ("vary_R", radius_sweep)):
        for path in sorted(p for p in sweep.iterdir() if (p / "metadata.json").exists() and (p / "results.json").exists()):
            runs.append((series, path, *load(path)))
    return runs


def summarize(updates_sweep, radius_sweep, output):
    runs = collect(updates_sweep, radius_sweep)
    spectral = next(r for r in runs if r[0] == "vary_R" and r[2] == "rhizome")
    base_meta, base_result, base = spectral[3], spectral[4], spectral[5]
    rows = []
    for series, path, variant, meta, result, fields in runs:
        for key in SHARED:
            if meta["config"][key] != base_meta["config"][key]:
                raise ValueError(f"uncontrolled config {key}: {path}")
        if meta["cache"] != base_meta["cache"] or not np.array_equal(fields["truth"], base["truth"]):
            raise ValueError(f"different test data: {path}")
        updates = meta["config"]["updates"]
        radius = None if variant == "rhizome" else float(variant.removeprefix("rhizome_r"))
        rows.append({
            "series": "spectral" if radius is None else series, "run": str(path.relative_to(path.parent.parent)),
            "radius": radius, "updates": updates, "reach_cells": None if radius is None else radius * updates,
            "parameters_real": result["parameters_real"], "best_step": result["best_step"],
            "best_validation_bce": result["best_validation_bce"], "test": result["test"],
            "train_seconds_including_validation": result["train_seconds_including_validation"],
            "rmse_ratio_vs_spectral": result["test"]["all"]["rmse"] / base_result["test"]["all"]["rmse"],
            "rmse_ratio_cone_bootstrap_95": cone_bootstrap_ratio(
                fields["prediction"], base["prediction"], base["truth"], base["cone_id"]),
            "iteration_sweep": {k: v["all"]["rmse"] for k, v in result.get("iteration_sweep", {}).items()},
        })
    # R=2, T=4 lives in the radius sweep but belongs to both series.
    rows += [dict(r, series="vary_T") for r in rows if r["series"] == "vary_R" and r["radius"] == 2.0]
    rows.sort(key=lambda r: (r["series"], r["reach_cells"] or 0))
    summary = {"comparability_checks_passed": True, "reference": "spectral, T=4", "rows": rows,
               "interpretation": "One seed per configuration; intervals cover test-cone sampling only. "
                                 "Training times come from different parallel batches and are observational."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    figure(rows, output / "reach_r2_vs_radius.png")
    print(f"{'series':8s} {'R':>4s} {'T':>3s} {'reach':>5s} {'params':>6s} {'valBCE':>7s} {'RMSE':>7s} "
          f"{'RMSEmix':>7s} {'x spec':>6s} {'95% (cones)':>13s} {'step':>5s} {'min':>5s}")
    for r in rows:
        lo, hi = r["rmse_ratio_cone_bootstrap_95"]
        print(f"{r['series']:8s} {r['radius'] or 0:4.1f} {r['updates']:3d} {r['reach_cells'] or 0:5.0f} "
              f"{r['parameters_real']:6d} {r['best_validation_bce']:7.4f} {r['test']['all']['rmse']:7.4f} "
              f"{r['test']['mixed']['rmse']:7.4f} {r['rmse_ratio_vs_spectral']:6.3f} {lo:6.3f}-{hi:.3f} "
              f"{r['best_step']:5d} {r['train_seconds_including_validation'] / 60:5.1f}")
    return summary


def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, aqua, orange, ink, muted = "#2a78d6", "#1baf7a", "#eb6834", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted})
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    spectral = next(r for r in rows if r["series"] == "spectral")
    ax.axhline(spectral["test"]["all"]["rmse"], color=orange, lw=2, ls="--", label="spectral, 4 modes, T=4")
    for series, color, marker, name, key in (("vary_T", blue, "o", "R=2, vary updates T", "updates"),
                                             ("vary_R", aqua, "s", "T=4, vary radius R", "radius")):
        sel = sorted((r for r in rows if r["series"] == series), key=lambda r: r["reach_cells"])
        x = [r["reach_cells"] for r in sel]
        y = [r["test"]["all"]["rmse"] for r in sel]
        ax.plot(x, y, color=color, lw=2, marker=marker, ms=8, mec="white", mew=2, label=name)
        for r, xi, yi in zip(sel, x, y):
            text = f"T={r['updates']}" if key == "updates" else f"R={r['radius']:.3g}"
            ax.annotate(text, (xi, yi), textcoords="offset points", xytext=(6, 6 if key == "updates" else -14),
                        color=muted, fontsize=8.5)
    ax.set_xlabel("total reach R·T [cells]  (4 Mpc cells, 32-cell periodic grid)")
    ax.set_ylabel("test x_HI RMSE, all slices")
    ax.set_title("Same reach: more updates of a small kernel vs one larger kernel", color=ink, loc="left")
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--updates-sweep", type=Path, required=True)
    parser.add_argument("--radius-sweep", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.updates_sweep, args.radius_sweep, args.out)


if __name__ == "__main__":
    main()
