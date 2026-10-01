"""Train the 3-D rhizome operator on native LOS windows of fno-21cm's preparation.

    python -m ebm21cm.train_rhizome3d --run-dir runs/rhizome3d/w48 --width 48 --rank 48

Data, splits, normalization, window sampling, tiled full-cone inference and
metrics are fno-21cm's own (``dataset.los_windows``, ``fno_multifield``),
imported from ``--fno-root``, so results are directly comparable with its
LOS-window runs (same 2000-cone preparation and test cones). Only the model,
objective and schedule differ:

* x_HI by BCE on the window core (the halo is context, as in fno-21cm); with
  ``--targets neutral_fraction,brightness_temp`` also the normalized T_b by
  MSE (weight ``--tb-weight``), from a plain or physics-structured head;
* optional fno-21cm emulated global-history input channels (``--history-emulator``);
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


def masked_mse(pred, target, mask):
    loss = (pred - target) ** 2
    mask = mask.expand_as(loss).to(loss.dtype)
    return (loss * mask).sum() / mask.sum()


# Options added for multi-field runs. At these defaults they are left out of the
# resume signature, so x_HI-only runs started before they existed still resume.
MULTIFIELD_DEFAULTS = {"inputs": "density", "targets": "neutral_fraction", "history_emulator": "",
                       "tb_head": "plain", "tb_weight": 1.0, "monitor": "neutral_fraction"}


def structured_config(mapping, normalization, parameter_normalization, param_names):
    """Channel indices and statistics for StructuredBrightness3d, as fno-21cm derives them."""
    for name in ("density", "los_velocity"):
        if name not in mapping.inputs:
            raise ValueError(f"the structured T_b head needs {name} as an input")
    if list(parameter_normalization.names) != list(param_names):
        raise ValueError("parameter normalization order differs from the input channels")
    n_in = len(mapping.inputs)
    omm = list(param_names).index("OMm")
    return {"indices": {"density": mapping.inputs.index("density"), "velocity": mapping.inputs.index("los_velocity"),
                        "z": n_in, "omm": n_in + 1 + omm, "relative": n_in + 1 + len(param_names)},
            "stats": {"density_offset": normalization["density"]["offset"],
                      "density_scale": normalization["density"]["scale"],
                      "velocity_offset": normalization["los_velocity"]["offset"],
                      "velocity_scale": normalization["los_velocity"]["scale"],
                      "tb_offset": normalization["brightness_temp"]["offset"],
                      "tb_scale": normalization["brightness_temp"]["scale"],
                      "omm_mean": float(parameter_normalization.mean[omm]),
                      "omm_std": float(parameter_normalization.std[omm])}}


def install_histories(fm, dataset, spec, fno_root):
    """Append fno-21cm emulated global-history channels; returns their descriptions."""
    described = []
    for path in [q for q in spec.replace(":", ",").split(",") if q]:
        path = Path(path) if Path(path).is_absolute() else Path(fno_root) / path
        emulator = fm.HistoryEmulator(str(path))
        dataset.install_history(emulator)
        described.append(emulator.describe())
    return described


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
    p.add_argument("--init-from", default=None,
                   help="warm start a NEW run from another run's last.pt/best.pt weights (same model config)")
    p.add_argument("--init-optimizer", action="store_true",
                   help="with --init-from and a last.pt: also load its Adam state")
    p.add_argument("--inputs", default=MULTIFIELD_DEFAULTS["inputs"], help="input fields, e.g. density,los_velocity")
    p.add_argument("--targets", default=MULTIFIELD_DEFAULTS["targets"],
                   help="neutral_fraction, optionally followed by ,brightness_temp")
    p.add_argument("--history-emulator", default=MULTIFIELD_DEFAULTS["history_emulator"],
                   help="comma-separated fno-21cm global-history emulators (relative to --fno-root)")
    p.add_argument("--tb-head", choices=("plain", "structured"), default=MULTIFIELD_DEFAULTS["tb_head"])
    p.add_argument("--tb-weight", type=float, default=MULTIFIELD_DEFAULTS["tb_weight"],
                   help="weight of the normalized T_b MSE relative to the x_HI BCE")
    p.add_argument("--monitor", choices=("neutral_fraction", "mean"), default=MULTIFIELD_DEFAULTS["monitor"],
                   help="model selection: x_HI RMSE, or the weighted mean normalized MSE over targets (fno-21cm)")
    args = p.parse_args(argv)

    fm, FieldMapping, FieldRegistry, LOSWindowConfig, LOSWindowDataset = fno_pipeline(args.fno_root)
    preparation_path = Path(args.preparation or Path(args.fno_root) / "experiments/los_windows/preparation_xhi_2000.json")
    preparation = fm.read_json(preparation_path)
    registry = FieldRegistry.from_dict(preparation["registry"])
    mapping = FieldMapping.create(args.inputs.split(","), args.targets.split(","), preparation["conditioning"], registry)
    if mapping.targets[0] != "neutral_fraction":
        raise ValueError("the first target must be neutral_fraction")
    multifield = "brightness_temp" in mapping.targets
    dataset, rows, _ = fm.prepared_dataset(preparation, mapping)
    histories = install_histories(fm, dataset, args.history_emulator, args.fno_root)
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
    if multifield:  # x_HI-only configs stay exactly as before
        from dataset.lightcone_params import PARAM_NAMES

        model_config.update({"targets": list(mapping.targets), "tb_head": args.tb_head,
                             "structured": structured_config(mapping, dataset.normalization,
                                                             dataset.parameter_normalization, PARAM_NAMES)
                             if args.tb_head == "structured" else None})
    model = RhizomeOperator3d(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_windows = LOSWindowDataset(dataset, rows["train"], window, args.seed, augment=args.augment)
    steps_per_epoch = len(train_windows)
    total_steps = args.epochs * steps_per_epoch
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    signature = {k: v for k, v in vars(args).items()
                 if k not in ("max_hours", "max_steps", "save_minutes", "skip_final_eval", "workers")}
    # Unused warm-start options stay out of the signature, so runs started before
    # they existed still resume.
    if not args.init_from:
        signature.pop("init_from", None)
        signature.pop("init_optimizer", None)
    for key, default in MULTIFIELD_DEFAULTS.items():
        if signature.get(key) == default:
            signature.pop(key)
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
        warm = None
        if args.init_from:
            warm = torch.load(args.init_from, map_location="cpu", weights_only=False)
            if warm["model_config"] != model_config:
                raise ValueError("--init-from model config differs from the requested model")
            model.load_state_dict(warm["model"])
            if args.init_optimizer:
                if "optimizer" not in warm:
                    raise ValueError("--init-optimizer needs a last.pt (best.pt holds no optimizer state)")
                optimizer.load_state_dict(warm["optimizer"])
            # The new schedule (warmup + cosine from --lr) replaces the loaded learning rate.
            for group in optimizer.param_groups:
                group["lr"] = args.lr
            print(f"Warm start from {args.init_from}"
                  f"{' (with optimizer state)' if args.init_optimizer else ''}", flush=True)
        root = Path(__file__).resolve().parent.parent
        metadata = {
            "signature": signature, "model_config": model_config,
            "parameters_real": sum(q.numel() * (2 if q.is_complex() else 1) for q in model.parameters()),
            "preparation": str(preparation_path.resolve()), "input_channels": list(dataset.channel_names),
            "window": window.to_dict(), "splits": {k: len(v) for k, v in rows.items()},
            "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "objective": ("BCE(x_HI) + tb_weight * MSE(normalized T_b) on window core (halo excluded)"
                          if multifield else "BCE on window core (halo excluded), x_HI only"),
            "mapping": {"inputs": list(mapping.inputs), "targets": list(mapping.targets),
                        "conditioning": preparation["conditioning"]},
            "history_emulator": histories or None, "history_spec": args.history_emulator or None,
            "monitor": args.monitor, "tb_weight": args.tb_weight if multifield else None,
            "git_commit": git_commit(root), "fno_git_commit": git_commit(Path(args.fno_root)),
            "torch": str(torch.__version__), "device": torch.cuda.get_device_name() if cuda else "cpu",
            "init_from": None if warm is None else {
                "path": str(Path(args.init_from).resolve()), "optimizer": args.init_optimizer,
                "source_state": {k: warm["state"][k] for k in ("epoch", "step", "best_val_rmse", "best_epoch")}
                if "state" in warm else {"epoch": warm.get("epoch")}},
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

    weights = {"neutral_fraction": 1.0, "brightness_temp": args.tb_weight}

    def validate():
        """x_HI metrics (keys as before), T_b metrics prefixed tb_, and the selection score."""
        result = fm.evaluate_rows(model, dataset, val_rows, device, 1, 0, 0, window)
        keys = ("rmse", "mae", "normalized_mse", "mean_bias", "pearson_r")
        val = {k: result["neutral_fraction"][k] for k in keys}
        if multifield:
            val.update({f"tb_{k}": result["brightness_temp"][k] for k in keys})
        if args.monitor == "mean":
            val["score"] = sum(result[n]["normalized_mse"] * weights[n] for n in mapping.targets) / \
                sum(weights[n] for n in mapping.targets)
        else:
            val["score"] = val["rmse"]
        return val

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
            out = model(x, logits=True)
            loss = masked_bce(out[:, :1], y[:, :1], mask)
            if multifield:
                loss = loss + args.tb_weight * masked_mse(out[:, 1:2], y[:, 1:2], mask)
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
                      f"{'loss' if multifield else 'bce'} {window_loss / window_n:.5f} lr {optimizer.param_groups[0]['lr']:.2e} "
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
        # "best_val_rmse" holds the selection score (x_HI RMSE unless --monitor mean).
        if val["score"] < state["best_val_rmse"]:
            state["best_val_rmse"], state["best_epoch"] = val["score"], state["epoch"]
            torch.save({"model": model.state_dict(), "model_config": model_config, "epoch": state["epoch"],
                        "val": val}, run_dir / "best.pt")
        tb = f" tb_rmse {val['tb_rmse']:.4f} tb_nmse {val['tb_normalized_mse']:.5f}" if multifield else ""
        sel = f" score {val['score']:.5f}" if args.monitor == "mean" else ""
        print(f"epoch {state['epoch']} done: val rmse {val['rmse']:.5f} nmse {val['normalized_mse']:.5f}{tb}{sel} "
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
        if (run_dir / name).exists():  # e.g. an evaluation interrupted between the two splits
            print(f"{split}: {name} exists, skipped", flush=True)
            continue
        result = fm.evaluate_rows(model, dataset, rows[split], device, 1, 0, 12, window)
        (run_dir / name).write_text(json.dumps({"checkpoint": "best.pt", "epoch": best["epoch"], "split": split,
                                                "fields": result}, indent=2, default=float))
        print(f"{split}: rmse {result['neutral_fraction']['rmse']:.5f}"
              + (f" tb_rmse {result['brightness_temp']['rmse']:.4f}" if multifield else ""), flush=True)


if __name__ == "__main__":
    main()
