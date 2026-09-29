"""Rank train_rhizome runs under a directory by validation scores only.

    python -m ebm21cm.sweep_summary runs/sweep [--sort bce|mixed]

Reads each run's ``metrics.jsonl`` and ``metadata.json``; never reads test
fields. ``mixed`` is the mean per-slice x_HI RMSE over validation slices with
0.05 < mean x_HI < 0.95 at the best-BCE check (fno-21cm's val_l2 definition).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(run_dir: Path):
    metrics, metadata = run_dir / "metrics.jsonl", run_dir / "metadata.json"
    if not metrics.exists() or not metadata.exists():
        return None
    history = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    meta = json.loads(metadata.read_text())
    checks = [r for r in history if r["step"] > 0]
    if not checks:
        return None
    best = min(checks, key=lambda r: r["validation_bce"])
    config = meta["config"]
    return {
        "run": run_dir.name, "step": history[-1]["step"], "best_step": best["step"],
        "bce": best["validation_bce"], "mixed": best.get("validation_mixed_slice_rmse"),
        "min_mixed": min((r["validation_mixed_slice_rmse"] for r in checks
                          if r.get("validation_mixed_slice_rmse") is not None), default=None),
        "train_probe": best.get("train_probe_bce"), "lr": history[-1]["next_lr"],
        "config": {k: config[k] for k in ("width", "modes", "updates", "step_size", "lr", "batch_size", "channels")
                   if k in config},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root")
    parser.add_argument("--sort", choices=("bce", "mixed"), default="bce")
    args = parser.parse_args(argv)
    rows = [r for d in sorted(Path(args.root).iterdir()) if d.is_dir() and (r := summarize(d))]
    rows.sort(key=lambda r: (r[args.sort] is None, r[args.sort] or 0.0))
    print(f"{'run':24s} {'step':>6s} {'best':>6s} {'val_bce':>9s} {'mixed':>8s} {'min_mix':>8s} "
          f"{'probe':>8s} {'lr':>8s}  config")
    for r in rows:
        fmt = lambda v: f"{v:8.5f}" if v is not None else f"{'-':>8s}"
        print(f"{r['run']:24s} {r['step']:6d} {r['best_step']:6d} {r['bce']:9.5f} {fmt(r['mixed'])} "
              f"{fmt(r['min_mixed'])} {fmt(r['train_probe'])} {r['lr']:8.1e}  {json.dumps(r['config'])}")


if __name__ == "__main__":
    main()
