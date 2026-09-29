"""Plot a trained rhizome x_HI model's predictions next to the ground truth.

    python -m ebm21cm.plot_rhizome --run-dir runs/sweep1/center --split validation --n 8

Picks ``--n`` slices from distinct cones whose true mean x_HI is spread evenly
over (``--lo``, ``--hi``), plus ``--neutral`` early-phase slices above ``--hi``,
predicts them with ``best.pt`` and writes ``predictions_<split>.png`` and a
per-slice JSON to the run directory. The test split is refused unless
``--allow-test`` is given, so development plots never touch held-out cones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .data.cache import SPLITS
from .data.memory import MemorySplit
from .train_recurrent import load_checkpoint
from .viz import regression_examples


def pick_rows(cache, split, n, lo, hi, neutral, seed=0):
    """Cache rows at evenly spaced true mean x_HI levels, at most one per cone."""
    with h5py.File(cache, "r") as h:
        rows = np.flatnonzero(h["split"][:] == SPLITS[split])
        cone = h["cone_id"][:][rows]
        means = np.array([h["xhi"][r].astype(np.float64).mean() for r in rows])
    rng = np.random.default_rng(seed)
    picked, used = [], set()
    levels = list(np.linspace(lo, hi, n)) + list(np.linspace(hi, 1.0, neutral + 2)[1:-1])
    for level in levels:
        order = np.argsort(np.abs(means - level) + 1e-6 * rng.random(len(means)))
        for j in order:
            if cone[j] not in used:
                picked.append(int(rows[j]))
                used.add(cone[j])
                break
    return picked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cache", help="defaults to the cache recorded in metadata.json")
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--lo", type=float, default=0.1)
    parser.add_argument("--hi", type=float, default=0.9)
    parser.add_argument("--neutral", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        parser.error("plotting test cones requires --allow-test")
    run_dir = Path(args.run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    cache = args.cache or metadata["cache"]
    model, checkpoint = load_checkpoint(run_dir / "best.pt", args.device)
    rows = pick_rows(cache, args.split, args.n, args.lo, args.hi, args.neutral)
    with h5py.File(cache, "r") as h:
        cones = sorted({int(h["cone_id"][r]) for r in rows})
    data = MemorySplit(cache, args.split, checkpoint["stats"], checkpoint.get("channels", "all"),
                       cones=cones, device=args.device)
    positions = [int(np.flatnonzero(data.rows == r)[0]) for r in rows]
    batch = data.batch(positions)
    with torch.no_grad():
        pred = model(batch["cond"], batch["scalars"]).sigmoid()[:, 0].cpu().numpy()
    truth = ((batch["target"][:, 0] + 1) / 2).cpu().numpy()
    order = np.argsort(truth.mean((1, 2)))
    density = data.delta[positions][:, data.center_band].float().numpy()
    z = batch["z"].numpy()
    out = Path(args.out or run_dir / f"predictions_{args.split}.png")
    title = (f"{run_dir.name}: step {checkpoint['step']}, val BCE {checkpoint['validation_bce']:.4f} "
             f"({args.split} cones)")
    regression_examples(out, density[order], truth[order], pred[order], z[order], title)
    report = [{"row": rows[i], "cone_id": int(batch["cone_id"][i]), "z": float(z[i]),
               "mean_truth": float(truth[i].mean()), "mean_pred": float(pred[i].mean()),
               "rmse": float(np.sqrt(np.mean((pred[i] - truth[i]) ** 2)))} for i in order]
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
