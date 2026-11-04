"""Strict comparison of two LoReLi eval output directories.

    python scripts/compare_loreli_outputs.py --old runs/eq_old --new runs/eq_new --sims 12280 12608 \
        --skip-shape \
        --ignore-keys n_predicted,slice_stride,min_per_stage,shape_per_stage,baseline_per_stage,no_baselines,z_window,device,torch_inputs,ids,limit,code_commit,code_dirty,n_targets,seconds

Exit status is nonzero on a missing key, a list-length change, a one-NaN/one-finite pair, an HDF5 mask difference
(including `valid`/`predicted`), a record or stage-count difference, or a numeric difference above the thresholds.
Old stage names early/mid/late are normalised to the xHI_0.05-0.35 / xHI_0.35-0.65 / xHI_0.65-0.95 bins.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

STAGE_ALIASES = {"early": "xHI_0.05-0.35", "mid": "xHI_0.35-0.65", "late": "xHI_0.65-0.95"}
SHAPE_KEYS = ("bubble_w1_", "front_width_", "tb_pdf_w1", "xhi_rmse_curve")


def normalize(record):
    record = dict(record)
    record["stages"] = {STAGE_ALIASES.get(k, k): v for k, v in record.get("stages", {}).items()}
    if "n_active" in record:
        record["n_active"] = {STAGE_ALIASES.get(k, k): v for k, v in record["n_active"].items()}
    return record


def compare(a, b, path, *, skip_shape, ignore_top, metric_tol, failures, diffs, top=True):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) - set(b)):
            if top and key in ignore_top:
                continue
            failures.append(f"{path}.{key}: missing in new")
        for key in sorted(set(b) - set(a)):
            if top and key in ignore_top:
                continue
            failures.append(f"{path}.{key}: extra in new")
        for key in sorted(set(a) & set(b)):
            if top and key in ignore_top:
                continue
            if skip_shape and any(key.startswith(prefix) for prefix in SHAPE_KEYS):
                continue
            compare(a[key], b[key], f"{path}.{key}" if path else key, skip_shape=skip_shape,
                    ignore_top=ignore_top, metric_tol=metric_tol, failures=failures, diffs=diffs, top=False)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            failures.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", skip_shape=skip_shape, ignore_top=ignore_top,
                    metric_tol=metric_tol, failures=failures, diffs=diffs, top=False)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            failures.append(f"{path}: {a} != {b}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            if not (math.isnan(fa) and math.isnan(fb)):
                failures.append(f"{path}: one NaN ({a} vs {b})")
            return
        diff = abs(fa - fb)
        diffs.append((path, diff))
        if diff > metric_tol:
            failures.append(f"{path}: {diff:.3e} > {metric_tol:.3e} ({a} vs {b})")
    elif a != b:
        failures.append(f"{path}: {a!r} != {b!r}")


def compare_h5(old, new, sim, *, xhi_tol, tb_tol, failures, diffs):
    paths = [old / f"cone_{sim}.h5", new / f"cone_{sim}.h5"]
    if not all(p.exists() for p in paths):
        failures.append(f"cone {sim}: h5 missing")
        return
    with h5py.File(paths[0], "r") as a, h5py.File(paths[1], "r") as b:
        for key in ("xhi", "tb"):
            if key not in a or key not in b:
                failures.append(f"cone {sim}: h5 {key} missing")
                continue
            x, y = a[key][:].astype(np.float64), b[key][:].astype(np.float64)
            if x.shape != y.shape:
                failures.append(f"cone {sim}: {key} shape {x.shape} != {y.shape}")
                continue
            finite = np.isfinite(x) & np.isfinite(y)
            only = np.isfinite(x) ^ np.isfinite(y)
            if only.any():
                failures.append(f"cone {sim}: {key} finite mask differs in {int(only.sum())} cells")
            diff = np.abs(x[finite] - y[finite])
            value = float(diff.max()) if diff.size else 0.0
            diffs.append((f"cone {sim}.{key}", value))
            tol = xhi_tol if key == "xhi" else tb_tol
            if value > tol:
                failures.append(f"cone {sim}: {key} max abs diff {value:.3e} > {tol:.3e}")
        for key in ("valid", "predicted"):
            if key in a and key in b:
                if not np.array_equal(a[key][:], b[key][:]):
                    failures.append(f"cone {sim}: {key} mask differs")
            elif key in a or key in b:
                if key == "predicted":
                    continue  # 3a623b4 has no predicted mask; the finite mask above still covers it
                failures.append(f"cone {sim}: {key} missing in one file")
        if "z" in a and "z" in b and not np.allclose(a["z"][:], b["z"][:]):
            failures.append(f"cone {sim}: z axis differs")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--sims", type=int, nargs="+", required=True)
    ap.add_argument("--xhi-tol", type=float, default=1e-3)
    ap.add_argument("--tb-tol", type=float, default=1e-2, help="mK; float16 storage")
    ap.add_argument("--metric-tol", type=float, default=1e-5)
    ap.add_argument("--skip-shape", action="store_true", help="skip bubble/front/PDF/curve metrics (old 48-subset)")
    ap.add_argument("--ignore-keys", default="seconds",
                    help="comma-separated top-level record keys not to compare (provenance additions)")
    args = ap.parse_args(argv)
    ignore = {s for s in args.ignore_keys.split(",") if s}
    failures, diffs = [], []
    for sim in args.sims:
        old_json = args.old / f"cone_{sim}.json"
        new_json = args.new / f"cone_{sim}.json"
        if not old_json.exists() or not new_json.exists():
            failures.append(f"cone {sim}: JSON missing")
            continue
        old = normalize(json.loads(old_json.read_text()))
        new = normalize(json.loads(new_json.read_text()))
        compare(old, new, "", skip_shape=args.skip_shape, ignore_top=ignore, metric_tol=args.metric_tol,
                failures=failures, diffs=diffs)
        compare_h5(args.old, args.new, sim, xhi_tol=args.xhi_tol, tb_tol=args.tb_tol,
                   failures=failures, diffs=diffs)
        metric_max = max((d for p, d in diffs if not p.endswith((".xhi", ".tb"))), default=0.0)
        field_max = {p.rsplit(".", 1)[-1]: d for p, d in diffs if p.endswith((".xhi", ".tb"))}
        print(f"cone {sim}: max |Delta x_HI| = {field_max.get('xhi', 0.0):.3e}, "
              f"max |Delta T_b| = {field_max.get('tb', 0.0):.3e}, max |Delta metric| = {metric_max:.3e}")
    if failures:
        print(f"\\nFAIL ({len(failures)}):")
        for failure in failures[:50]:
            print(f"  {failure}")
        raise SystemExit(1)
    print("\\nPASS: no missing keys, length changes, mask differences or values above threshold.")


if __name__ == "__main__":
    main()
