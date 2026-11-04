"""Evaluate LoReLi checkpoints on one split of a LoReLi slice cache (default: test).

Works for fine-tuned runs (``best.pt`` saved on the 2x LoReLi domain) and for the
21cmFAST reference run (remapped on load), so "before fine-tuning" is scored on
exactly the same rows.  Metrics are those used for selection during fine-tuning,
on the valid 206x206 interior: BCE, x_HI RMSE, T_b RMSE (mK), score = mean of the
normalized x_HI and T_b MSEs - overall and per reionization stage of the target
slice (``stage`` column of the cache).

History scalars: ``--scalars-file`` (e.g. the emulated-history side file); by
default the one the run was trained with (``run_metadata.json``), else the cache's
own (oracle) histories.  ``--scalars-file none`` forces the oracle.

    python -m ebm21cm.evaluate_loreli_finetune --run runs/loreli_finetune/<run> \
        --cache .../loreli_slices_l294_v3.h5 --split test
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from .data.loreli_memory import LoReLiMemorySplit
from .train_loreli_finetune import _batch_loss, load_reference

STAGES = {-1: "outside stage bins", 0: "xHI_0.05-0.35", 1: "xHI_0.35-0.65", 2: "xHI_0.65-0.95"}


def _scalars_file(run: Path, override: str | None):
    if override is not None:
        return None if override.lower() == "none" else override
    meta = run / "run_metadata.json"
    if meta.exists():
        value = json.loads(meta.read_text()).get("config", {}).get("scalars_file") or None
        return value
    return None


@torch.no_grad()
def evaluate_run(run, cache, split="test", scalars_file=None, device="cuda", batch_size=8, blocks=None,
                 max_rows=None, log=print):
    run = Path(run)
    device = torch.device(device)
    model, _, best = load_reference(run)
    model = model.to(device).eval()
    tb_norm = tuple(float(v) for v in best["tb_norm"])
    keep = best.get("scalar_keep")
    data = LoReLiMemorySplit(cache, split, blocks=blocks, storage="cpu", device=device, scalars_file=scalars_file)
    if keep is not None:
        data.scalars = data.scalars[:, torch.as_tensor(keep, dtype=torch.long)]
    with h5py.File(cache, "r") as h:
        stage = h["stage"][:][data.rows]
    n_rows = len(data) if not max_rows else min(len(data), int(max_rows))
    stage = stage[:n_rows]
    per_row = {k: np.zeros(n_rows) for k in ("n", "bce", "x_se", "tb_se")}
    for a in range(0, n_rows, batch_size):
        positions = torch.arange(a, min(a + batch_size, n_rows))
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.batch(positions).items()}
        _, out, bce, tb_mse, mask, target = _batch_loss(model, batch, tb_norm, 1.0)
        dims = (1, 2, 3)
        per_row["n"][a:a + len(positions)] = mask.sum(dims).cpu().numpy()
        per_row["bce"][a:a + len(positions)] = (bce * mask).sum(dims).cpu().numpy()
        per_row["x_se"][a:a + len(positions)] = (((out[:, :1].sigmoid() - target[:, :1]) ** 2) * mask).sum(dims).cpu().numpy()
        per_row["tb_se"][a:a + len(positions)] = (tb_mse * mask).sum(dims).cpu().numpy()

    def summarize(sel):
        n = per_row["n"][sel].sum()
        x_mse, tb_mse = per_row["x_se"][sel].sum() / n, per_row["tb_se"][sel].sum() / n
        return {"rows": int(sel.sum()), "bce": float(per_row["bce"][sel].sum() / n),
                "xhi_rmse": math.sqrt(x_mse), "tb_rmse_mk": math.sqrt(tb_mse) * tb_norm[1],
                "score": 0.5 * (x_mse + tb_mse)}

    result = {"run": str(run.resolve()), "cache": str(Path(cache).resolve()), "split": split,
              "scalars_file": scalars_file, "histories": "side file" if scalars_file else "cache (oracle)",
              "checkpoint_step": best.get("step"), "all": summarize(np.ones(n_rows, bool)),
              "stages": {STAGES[s]: summarize(stage == s) for s in sorted(STAGES) if (stage == s).any()}}
    log(json.dumps(result, indent=1))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ebm21cm.evaluate_loreli_finetune")
    p.add_argument("--run", required=True, nargs="+", help="run directories (best.pt)")
    p.add_argument("--cache", required=True)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--scalars-file", default=None, help="side file, or 'none' for the cache's oracle histories")
    p.add_argument("--out-name", default=None, help="JSON written into each run dir (default <split>_metrics.json)")
    p.add_argument("--out", default=None, help="explicit JSON path (single run only)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--blocks", type=int, nargs="*", default=None, help="restrict to these blocks (smoke tests)")
    p.add_argument("--max-rows", type=int, default=None, help="evaluate only the first N rows (smoke tests)")
    a = p.parse_args(argv)
    for run in a.run:
        run = Path(run)
        side = _scalars_file(run, a.scalars_file)
        result = evaluate_run(run, a.cache, a.split, side, a.device, a.batch_size, a.blocks, a.max_rows,
                              log=lambda s: print(s, flush=True))
        target = Path(a.out) if a.out else run / (a.out_name or f"{a.split}_metrics.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
