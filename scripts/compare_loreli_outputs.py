"""Compare two LoReLi eval output directories (old vs new) for the equality check.

    python scripts/compare_loreli_outputs.py --old runs/eq_old --new runs/eq_new --sims 12280 12608

Reports the maximum absolute difference of the stored x_HI and T_b predictions
(float16, NaN outside the evaluated window) and of every numeric metric in the
per-cone JSON. Differences are computed only where both values are finite.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


def numeric_diff(a, b, path="", out=None):
    """Collect (path, |a-b|) for numeric leaves; NaN in both counts as zero."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) & set(b)):
            numeric_diff(a[key], b[key], f"{path}.{key}" if path else key, out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        for i, (x, y) in enumerate(zip(a, b)):
            numeric_diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, bool) or isinstance(b, bool):
            return out
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            out.append((path, 0.0))
        elif math.isfinite(fa) and math.isfinite(fb):
            out.append((path, abs(fa - fb)))
    return out


def compare_h5(old, new, sim):
    with h5py.File(old / f"cone_{sim}.h5", "r") as a, h5py.File(new / f"cone_{sim}.h5", "r") as b:
        result = {}
        for key in ("xhi", "tb"):
            x = a[key][:].astype(np.float64)
            y = b[key][:].astype(np.float64)
            both = np.isfinite(x) & np.isfinite(y)
            only = np.isfinite(x) ^ np.isfinite(y)
            diff = np.abs(x[both] - y[both])
            result[key] = {"max_abs": float(diff.max()) if diff.size else 0.0,
                           "n_finite_only_one": int(only.sum())}
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--sims", type=int, nargs="+", required=True)
    args = ap.parse_args(argv)
    for sim in args.sims:
        h5 = compare_h5(args.old, args.new, sim)
        ja = json.loads((args.old / f"cone_{sim}.json").read_text())
        jb = json.loads((args.new / f"cone_{sim}.json").read_text())
        diffs = numeric_diff(ja.get("stages", {}), jb.get("stages", {}))
        max_metric = max((d for _, d in diffs), default=0.0)
        worst = sorted(diffs, key=lambda kv: kv[1], reverse=True)[:5]
        print(f"cone {sim}:")
        print(f"  max |Delta x_HI| = {h5['xhi']['max_abs']:.3e}  (finite-only-one: {h5['xhi']['n_finite_only_one']})")
        print(f"  max |Delta T_b|  = {h5['tb']['max_abs']:.3e}  (finite-only-one: {h5['tb']['n_finite_only_one']})")
        print(f"  max |Delta metric| = {max_metric:.3e}")
        for name, value in worst:
            if value > 0:
                print(f"    {name}: {value:.3e}")


if __name__ == "__main__":
    main()
