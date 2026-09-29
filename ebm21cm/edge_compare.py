"""Boundary-band edge metrics: rhizome vs fno-21cm 2-D models on identical slices.

    python -m ebm21cm.edge_compare --rhizome runs/sweep1/w128 \\
        --fno LWF=$FNO/checkpoints/2d_xhi/fno_lwf/xhi2d_lwf_glob_e100 --out runs/edge_fno/w128

Scores are fno-21cm's own ``viz.boundary_band_diagnostic`` in its 2-D ``slice``
mode (as ``viz.edge_metrics_xhi2d`` uses it): errors binned by the signed
distance to the true x_HI=0.5 front, boundary-band L2 and H1 (gradient) RMS at
+-2 and +-5 Mpc, and the 10-90 front width of the mean x_HI profile across the
front. Slices and inputs are those of :mod:`ebm21cm.compare_fno`. Differences
to the first ``--fno`` model get a paired bootstrap over cones.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from .compare_fno import fno_modules, predict_fno, predict_rhizome, select_rows
from .data.memory import MemorySplit
from .train_recurrent import load_checkpoint as load_rhizome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rhizome", required=True)
    parser.add_argument("--rhizome-label", default="Rhizome")
    parser.add_argument("--fno", nargs="+", required=True, help="LABEL=fno-21cm 2-D checkpoint dir")
    parser.add_argument("--fno-root", default="/pfs/10/work/hd_id260-fno_training/fno-21cm")
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--window", type=float, nargs=2, default=(0.05, 0.95))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        parser.error("the rhizome test split requires --allow-test")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    load_run, build_model, load_fno_checkpoint, norm_cls = fno_modules(args.fno_root)
    from viz.boundary_band_diagnostic import (
        BandConfig, BoundaryBandAccumulator, front_width, paired_bootstrap, plot_overlay,
        saturation_calibration, write_csv,
    )

    runs = []
    for spec in args.fno:
        label, path = spec.split("=", 1)
        run = load_run(label, Path(path))
        if run is None:
            raise SystemExit(f"cannot load fno run {spec}")
        runs.append(run)
    rhizome_dir = Path(args.rhizome)
    meta = json.loads((rhizome_dir / "metadata.json").read_text())
    cache = meta["cache"]
    cones = set(meta["splits"][args.split]["cones"])
    for run in runs:
        cones &= set(run.metadata["split"]["test_cone_ids"])
    rows, _ = select_rows(cache, args.split, cones, args.window)
    _, checkpoint = load_rhizome(rhizome_dir / "best.pt", "cpu")
    data = MemorySplit(cache, args.split, checkpoint["stats"], checkpoint.get("channels", "all"),
                       cones=sorted(cones), storage="cpu", device=args.device)
    positions = np.searchsorted(data.rows, rows)
    with h5py.File(cache, "r") as h:
        cell = float(h.attrs["cell_size_mpc"])
        bands = json.loads(h.attrs["bands"])
        center = [j for j, (lo, hi) in enumerate(bands) if lo == 0 and hi == 1][0]
        density = np.stack([h["delta"][r, center] for r in rows]).astype(np.float32)
        truth = np.stack([h["xhi"][r] for r in rows]).astype(np.float64)
        z = h["z"][:][rows]
        params = h["params"][:][rows]
        cone_id = h["cone_id"][:][rows]
    print(f"{len(rows)} slices from {len(cones)} cones held out by every model", flush=True)

    preds = {args.rhizome_label: predict_rhizome(rhizome_dir, data, positions, args.device, args.batch)[0]}
    for run in runs:
        preds[run.label] = predict_fno(run, build_model, load_fno_checkpoint, norm_cls, density, z, params,
                                       args.device, args.batch)

    cfg = BandConfig(mode="slice", threshold=args.threshold, dx_mpc=cell, dy_mpc=cell)
    rows_by_cone = defaultdict(list)
    for i, c in enumerate(cone_id):
        rows_by_cone[int(c)].append(i)
    results, accums, summary = {}, {}, {}
    for name, pred in preds.items():
        acc = BoundaryBandAccumulator(cfg=cfg)
        for cone, idx in rows_by_cone.items():
            # Slice mode treats each plane of the stack independently.
            acc.add_cone(cone, np.stack([pred[i] for i in idx], -1), np.stack([truth[i] for i in idx], -1))
        prof = acc.profile()
        band = acc.band_summary()
        d = np.asarray(prof["d_mpc"])
        results[name] = {"profile": prof, **band,
                         "front_width_truth_mpc": front_width(d, np.asarray(prof["mean_truth"])),
                         "front_width_pred_mpc": front_width(d, np.asarray(prof["mean_pred"]))}
        accums[name] = acc
        summary[name] = {**{k: v for k, v in results[name].items() if k != "profile"},
                         "total_sq_err": acc.total_sq_err,
                         **saturation_calibration(pred, truth)}
        print(f"{name:10s} front width {results[name]['front_width_pred_mpc']:6.2f} Mpc "
              f"(truth {results[name]['front_width_truth_mpc']:.2f})  "
              + "  ".join(f"{k} {v[0]:.4f}" for k, v in band.items() if not k.startswith("errfrac")), flush=True)

    # Paired bootstrap over cones: rhizome minus each fno model (negative = rhizome better).
    names = list(preds)
    per_cone = {n: {r["cone_id"]: r for r in accums[n].per_cone} for n in names}
    shared = sorted(set.intersection(*(set(p) for p in per_cone.values())))
    keys = [k for k in accums[names[0]].per_cone[0] if k.startswith(("L2_band", "H1_band"))]
    bootstrap = {}
    for other in names[1:]:
        bootstrap[other] = {k: paired_bootstrap([per_cone[names[0]][c][k] for c in shared],
                                                [per_cone[other][c][k] for c in shared]) for k in keys}
    plot_overlay(results, cfg, out / "edge_band_overlay.png")
    write_csv(results, accums, out / "edge_band_metrics.csv")
    (out / "summary.json").write_text(json.dumps(
        {"slices": len(rows), "cones": len(cones), "window": list(args.window), "threshold": args.threshold,
         "rhizome_checkpoint": {"run": str(rhizome_dir.resolve()), "step": checkpoint["step"]},
         "models": summary, "paired_bootstrap_rhizome_minus": bootstrap}, indent=2, default=float) + "\n")
    for other, stats in bootstrap.items():
        print(f"{names[0]} - {other}: " + "  ".join(
            f"{k} {s['mean_diff']:+.4f} [{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}]" for k, s in stats.items()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
