"""Train the 3-D rhizome operator on native LOS windows of fno-21cm's preparation.

    python -m ebm21cm.train_rhizome3d --run-dir runs/rhizome3d/w48 --width 48 --rank 48

Data, splits, normalization, window sampling, tiled full-cone inference and
metrics are fno-21cm's own (``dataset.los_windows``, ``fno_multifield``),
imported from ``--fno-root``, so results are directly comparable with its
LOS-window runs (same 2000-cone preparation and test cones). Only the model,
objective and schedule differ:

* x_HI only, BCE on the window core (the halo is context, as in fno-21cm);
* Adam with linear warmup then cosine decay over the step budget;
* optional transverse dihedral + periodic-shift augmentation (exact symmetries);
* per-epoch validation on fno-21cm's stratified validation subset, with the
  best checkpoint selected by validation RMSE; full validation and test
  splits are scored once, after the budget, in fno-21cm's test_metrics format.

Runs are resumable at exact sample order: ``last.pt`` is written every
``--save-minutes`` and at each epoch end. Resubmit the same command to continue.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from .model.rhizome3d import RhizomeOperator3d
from .run import git_commit


class EpochSampler(Sampler):
    """Seeded per-epoch permutation that can start part-way through an epoch."""

    def __init__(self, n, seed):
        self.n, self.seed, self.epoch, self.start = n, seed, 0, 0

    def order(self):
        return torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed * 100003 + self.epoch))

    def __iter__(self):
        return iter(self.order()[self.start:].tolist())

    def __len__(self):
        return self.n - self.start


def fno_pipeline(root):
    sys.path.insert(0, str(Path(root).resolve()))
    from util.neuralop_setup import prefer_local_neuralop

    prefer_local_neuralop()
    import fno_multifield as fm
    from dataset.fields import FieldMapping, FieldRegistry
    from dataset.los_windows import LOSWindowConfig, LOSWindowDataset

    return fm, FieldMapping, FieldRegistry, LOSWindowConfig, LOSWindowDataset


def lr_at(step, total, warmup, peak, floor):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def masked_bce(logits, target, mask):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    mask = mask.expand_as(loss).to(loss.dtype)
    return (loss * mask).sum() / mask.sum()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--fno-root", default="/pfs/10/work/hd_id260-fno_training/fno-21cm")
    p.add_argument("--preparation", default=None,
                   help="fno-21cm preparation JSON (default: experiments/los_windows/preparation_xhi_2000.json)")
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--modes", type=int, nargs=3, default=(24, 24, 16))
    p.add_argument("--updates", type=int, default=6)
    p.add_argument("--step-size", type=float, default=0.5)
    p.add_argument("--los-pad", type=int, default=64)
    p.add_argument("--rank", type=int, default=0, help="factorized kernel rank; 0 = dense per-mode matrices")
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--window-size", type=int, default=256)
    p.add_argument("--window-halo", type=int, default=32)
    p.add_argument("--windows-per-cone", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min-lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--augment", choices=("none", "transverse"), default="transverse")
    p.add_argument("--val-cones", type=int, default=40)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-minutes", type=float, default=30.0)
    p.add_argument("--max-hours", type=float, default=0.0,
                   help="stop cleanly (resumable) after this much wall time; 0 = no limit")
    p.add_argument("--max-steps", type=int, default=0, help="stop cleanly (resumable) at this step; 0 = none")
    p.add_argument("--skip-final-eval", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    fm, FieldMapping, FieldRegistry, LOSWindowConfig, LOSWindowDataset = fno_pipeline(args.fno_root)
    preparation_path = Path(args.preparation or Path(args.fno_root) / "experiments/los_windows/preparation_xhi_2000.json")
    preparation = fm.read_json(preparation_path)
    registry = FieldRegistry.from_dict(preparation["registry"])
    mapping = FieldMapping.create(["density"], ["neutral_fraction"], preparation["conditioning"], registry)
    dataset, rows, _ = fm.prepared_dataset(preparation, mapping)
    norm = dataset.normalization["neutral_fraction"]
    if norm["offset"] != 0.0 or norm["scale"] != 1.0:
        raise ValueError("x_HI must be un-normalized (offset 0, scale 1) for a sigmoid output")
    window = LOSWindowConfig(mode="contiguous", size=args.window_size, halo=args.window_halo,
                             windows_per_cone=args.windows_per_cone)
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_config = {"in_channels": dataset.in_channels, "width": args.width, "modes": list(args.modes),
                    "n_steps": args.updates, "step_size": args.step_size, "los_pad": args.los_pad,
                    "rank": args.rank or None, "checkpoint": not args.no_checkpoint, "amp": not args.no_amp}
    model = RhizomeOperator3d(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_windows = LOSWindowDataset(dataset, rows["train"], window, args.seed, augment=args.augment)
    steps_per_epoch = len(train_windows)
    total_steps = args.epochs * steps_per_epoch
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    signature = {k: v for k, v in vars(args).items()
                 if k not in ("max_hours", "max_steps", "save_minutes", "skip_final_eval", "workers")}
    state = {"epoch": 0, "position": 0, "step": 0, "best_val_rmse": float("inf"), "best_epoch": None,
             "elapsed": 0.0, "history": []}
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck["signature"] != signature:
            raise ValueError("resume options differ from the saved run")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        state = ck["state"]
        torch.set_rng_state(ck["torch_rng"])
        if cuda:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        print(f"Resumed epoch {state['epoch']} position {state['position']} (step {state['step']})", flush=True)
    else:
        if any(run_dir.iterdir()):
            raise ValueError(f"{run_dir} is not empty and has no last.pt")
        root = Path(__file__).resolve().parent.parent
        metadata = {
            "signature": signature, "model_config": model_config,
            "parameters_real": sum(q.numel() * (2 if q.is_complex() else 1) for q in model.parameters()),
            "preparation": str(preparation_path.resolve()), "input_channels": list(dataset.channel_names),
            "window": window.to_dict(), "splits": {k: len(v) for k, v in rows.items()},
            "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "objective": "BCE on window core (halo excluded), x_HI only",
            "git_commit": git_commit(root), "fno_git_commit": git_commit(Path(args.fno_root)),
            "torch": str(torch.__version__), "device": torch.cuda.get_device_name() if cuda else "cpu",
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    val_rows, val_info = fm.validation_subset(dataset, rows["val"], args.val_cones, "neutral_fraction")
    print(f"params {sum(q.numel() * (2 if q.is_complex() else 1) for q in model.parameters()) / 1e6:.1f}M; "
          f"{steps_per_epoch} windows/epoch x {args.epochs} epochs; val subset {len(val_rows)} cones", flush=True)

    def save(path=last):
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "state": state, "model_config": model_config, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if cuda else []}, tmp)
        tmp.replace(path)

    def validate():
        result = fm.evaluate_rows(model, dataset, val_rows, device, 1, 0, 0, window)["neutral_fraction"]
        return {k: result[k] for k in ("rmse", "mae", "normalized_mse", "mean_bias", "pearson_r")}

    wall = time.monotonic()
    last_save = wall
    sampler = EpochSampler(steps_per_epoch, args.seed)
    stopped = False
    while state["epoch"] < args.epochs and not stopped:
        train_windows.set_epoch(state["epoch"])
        sampler.epoch, sampler.start = state["epoch"], state["position"]
        loader = DataLoader(train_windows, batch_size=1, sampler=sampler, num_workers=args.workers,
                            pin_memory=cuda, persistent_workers=False, prefetch_factor=4 if args.workers else None)
        model.train()
        window_loss, window_n, tick = 0.0, 0, time.monotonic()
        for batch in loader:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(state["step"], total_steps, args.warmup, args.lr, args.min_lr)
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["loss_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce(model(x, logits=True), y, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {state['step']}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            state["step"] += 1
            state["position"] += 1
            window_loss += float(loss.detach())
            window_n += 1
            if state["step"] % 200 == 0:
                rate = window_n / (time.monotonic() - tick)
                print(f"epoch {state['epoch']} {state['position']}/{steps_per_epoch} step {state['step']} "
                      f"bce {window_loss / window_n:.5f} lr {optimizer.param_groups[0]['lr']:.2e} "
                      f"{rate:.2f} win/s", flush=True)
                window_loss, window_n, tick = 0.0, 0, time.monotonic()
            now = time.monotonic()
            if now - last_save > 60 * args.save_minutes:
                state["elapsed"] += now - last_save
                last_save = now
                save()
            if (args.max_hours and now - wall > 3600 * args.max_hours) or \
                    (args.max_steps and state["step"] >= args.max_steps):
                stopped = True
                break
        if stopped:
            break
        val = validate()
        record = {"epoch": state["epoch"], "step": state["step"], **{f"val_{k}": v for k, v in val.items()}}
        state["history"].append(record)
        if val["rmse"] < state["best_val_rmse"]:
            state["best_val_rmse"], state["best_epoch"] = val["rmse"], state["epoch"]
            torch.save({"model": model.state_dict(), "model_config": model_config, "epoch": state["epoch"],
                        "val": val}, run_dir / "best.pt")
        print(f"epoch {state['epoch']} done: val rmse {val['rmse']:.5f} nmse {val['normalized_mse']:.5f} "
              f"(best {state['best_val_rmse']:.5f} @ {state['best_epoch']})", flush=True)
        (run_dir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in state["history"]))
        state["epoch"] += 1
        state["position"] = 0
        now = time.monotonic()
        state["elapsed"] += now - last_save
        last_save = now
        save()
    now = time.monotonic()
    state["elapsed"] += now - last_save
    save()
    if stopped:
        print(f"Stopped (time/step limit) at epoch {state['epoch']} position {state['position']}; resubmit to continue.",
              flush=True)
        return
    if args.skip_final_eval or (run_dir / "test_metrics.json").exists():
        return
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    for split, name in (("val", "val_full_metrics.json"), ("test", "test_metrics.json")):
        result = fm.evaluate_rows(model, dataset, rows[split], device, 1, 0, 12, window)
        (run_dir / name).write_text(json.dumps({"checkpoint": "best.pt", "epoch": best["epoch"], "split": split,
                                                "fields": result}, indent=2, default=float))
        print(f"{split}: rmse {result['neutral_fraction']['rmse']:.5f}", flush=True)


if __name__ == "__main__":
    main()
