"""Resumable rhizome training to an explicit validation-plateau criterion.

Unlike the short architecture comparison, this runner reduces the learning
rate on plateaus and evaluates test fields only after the stopping criterion
is met. Reaching max_steps is a budget stop, not a convergence claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, default_collate

from .data.cache import SliceDataset, read_stats
from .data.memory import MemorySplit, MemoryStream, parse_channels
from .model.rhizome import RhizomeOperator2d
from .run import git_commit
from .train_recurrent import (
    ExperimentConfig, _batch, _example_rows, _plot_examples, _sync,
    field_metrics, load_checkpoint, predict, validation_bce,
)


@dataclass
class LongRunConfig:
    max_steps: int = 20_000
    min_steps: int = 4_000
    batch_size: int = 16
    width: int = 16
    modes: int = 4
    updates: int = 4
    step_size: float = 0.5
    lr: float = 0.003
    min_lr: float = 0.00003
    lr_factor: float = 0.5
    lr_patience: int = 4
    stop_patience: int = 8
    relative_min_delta: float = 0.002
    val_every: int = 200
    train_probe_rows: int = 256
    seed: int = 0
    threads: int = 2
    # "stream": per-row HDF5 reads (SliceDataset). "memory": splits held in
    # memory and batched/augmented on the device (data.memory); required for
    # band subsets and fast GPU training on full-size slices.
    data: str = "stream"
    channels: str = "all"
    # Train on a fixed random subset of this many training cones (0 = all).
    # The subset depends only on the cache, not on the seed.
    train_cones: int = 0
    eval_batch_size: int = 0
    # With data="memory": where the splits live. "device" keeps them on the
    # training device; "cpu" keeps them in host RAM and copies each batch,
    # leaving the GPU to activations when the full training split is large.
    storage: str = "device"

    def validate(self):
        ExperimentConfig(steps=self.max_steps, batch_size=self.batch_size, width=self.width,
                         modes=self.modes, updates=self.updates, step_size=self.step_size,
                         lr=self.lr, val_every=self.val_every, seed=self.seed,
                         threads=self.threads).validate()
        for name in ("min_steps", "lr_patience", "stop_patience", "train_probe_rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_steps > self.max_steps:
            raise ValueError("min_steps cannot exceed max_steps")
        if not 0 < self.min_lr <= self.lr or not 0 < self.lr_factor < 1:
            raise ValueError("require 0 < min_lr <= lr and 0 < lr_factor < 1")
        if not 0 <= self.relative_min_delta < 1:
            raise ValueError("relative_min_delta must be in [0, 1)")
        if self.data not in ("stream", "memory"):
            raise ValueError("data must be 'stream' or 'memory'")
        if self.data == "stream" and (self.channels != "all" or self.train_cones):
            raise ValueError("channels and train_cones require data='memory'")
        if self.storage not in ("device", "cpu"):
            raise ValueError("storage must be 'device' or 'cpu'")
        for name in ("train_cones", "eval_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass
class PlateauState:
    best: float
    reference: float
    best_step: int = 0
    bad_checks: int = 0
    reductions: int = 0

    def observe(self, loss, step, lr, config):
        """Return (new LR, stop, new absolute best).

        Checkpoints use the absolute minimum, while plateau detection requires
        a relative improvement over the last *significant* reference. Smaller
        cumulative improvements can eventually reset the plateau counter.
        A rate reduction resets the counter, guaranteeing a fresh patience
        window at the new rate.
        """
        if not math.isfinite(loss):
            raise FloatingPointError("non-finite validation loss")
        improved = loss < self.best
        if improved:
            self.best, self.best_step = loss, step
        if loss < self.reference * (1 - config.relative_min_delta):
            self.reference, self.bad_checks = loss, 0
        else:
            self.bad_checks += 1
        if lr > config.min_lr * (1 + 1e-12) and self.bad_checks >= config.lr_patience:
            lr = max(config.min_lr, lr * config.lr_factor)
            self.bad_checks = 0
            self.reductions += 1
        stop = (step >= config.min_steps and lr <= config.min_lr * (1 + 1e-12)
                and self.bad_checks >= config.stop_patience)
        return lr, stop, improved


class TrainingStream:
    """Single-process shuffled batches with exact order/augmentation resumption."""

    def __init__(self, dataset, batch_size, seed):
        self.dataset, self.batch_size = dataset, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(dataset), generator=self.generator)
        self.position, self.epoch = 0, 0

    def next(self):
        if self.position == len(self.order):
            self.order = torch.randperm(len(self.dataset), generator=self.generator)
            self.position, self.epoch = 0, self.epoch + 1
        end = min(self.position + self.batch_size, len(self.order))
        rows = self.order[self.position:end].tolist()
        batch = default_collate([self.dataset[i] for i in rows])
        self.position = end
        return batch

    def state_dict(self):
        return {
            "order": self.order, "position": self.position, "epoch": self.epoch,
            "generator": self.generator.get_state(),
            "augmentation": self.dataset._rng.bit_generator.state if self.dataset._rng is not None else None,
        }

    def load_state_dict(self, state):
        self.order, self.position, self.epoch = state["order"], state["position"], state["epoch"]
        self.generator.set_state(state["generator"])
        self.dataset._rng = np.random.default_rng()
        if state["augmentation"] is not None:
            self.dataset._rng.bit_generator.state = state["augmentation"]
        else:
            self.dataset._rng = None


@torch.no_grad()
def validation_scores(model, loader, device):
    """Pixel-mean BCE and x_HI RMSE, plus the mean per-slice RMSE over mixed-phase
    slices (0.05 < mean x_HI < 0.95), which is fno-21cm's val_l2 on such slices."""
    model.eval()
    bce, squared, pixels, slice_rmse, mixed_slices = 0.0, 0.0, 0, 0.0, 0
    for batch in loader:
        cond, scalars, truth = _batch(batch, device)
        logits = model(cond, scalars)
        bce += float(F.binary_cross_entropy_with_logits(logits, truth, reduction="sum"))
        error = (logits.sigmoid() - truth).square()
        squared += float(error.sum())
        pixels += truth.numel()
        means = truth.mean((1, 2, 3))
        mixed = (means > 0.05) & (means < 0.95)
        slice_rmse += float(error[mixed].mean((1, 2, 3)).sqrt().sum())
        mixed_slices += int(mixed.sum())
    return {"bce": bce / pixels, "rmse": math.sqrt(squared / pixels),
            "mixed_slice_rmse": slice_rmse / mixed_slices if mixed_slices else None}


def train_cone_subset(cones, limit):
    """A fixed random subset of ``limit`` training cones, or None for all of them."""
    cones = np.array(sorted(cones))
    if not limit or limit >= len(cones):
        return None
    return np.sort(np.random.default_rng(0).permutation(cones)[:limit])


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _curves(path, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for key in ("train_window_bce", "train_probe_bce", "validation_bce"):
        rows = [r for r in history if r.get(key) is not None]
        axes[0].plot([r["step"] for r in rows], [r[key] for r in rows], label=key)
    axes[0].set_ylabel("BCE")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[1].step([r["step"] for r in history], [r["next_lr"] for r in history], where="post")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Learning rate")
    axes[1].set_xlabel("Optimizer step")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(cache_path, run_dir, config=None, device="cpu", resume=False):
    config = config or LongRunConfig()
    config.validate()
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("use cpu or cuda; complex FFTs are not validated on MPS")
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    run_dir = Path(run_dir)
    initial_metadata = None
    if resume and not (run_dir / "last.pt").exists():
        # No optimizer step can run before the first atomic checkpoint. Allow
        # a matching metadata-only setup to restart from its original seed.
        names = {p.name for p in run_dir.iterdir()} if run_dir.exists() else set()
        if "metadata.json" in names and names <= {"metadata.json", "last.pt.tmp"}:
            initial_metadata = json.loads((run_dir / "metadata.json").read_text())
            resume = False
        else:
            raise ValueError("--resume requires an existing last.pt")
    if not resume and initial_metadata is None and run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("run directory is not empty; use --resume or a new directory")
    stats = read_stats(cache_path)
    datasets = {s: SliceDataset(cache_path, s, stats) for s in ("train", "validation", "test")}
    if any(len(ds) == 0 for ds in datasets.values()):
        raise ValueError("all three cache splits must be nonempty")
    cones = {s: set(ds.cone_id.tolist()) for s, ds in datasets.items()}
    if any(cones[a] & cones[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("cone IDs overlap across splits")
    cache_hash = _sha256(cache_path)
    root = Path(__file__).resolve().parent.parent
    source_paths = ("ebm21cm/train_rhizome.py", "ebm21cm/train_recurrent.py", "ebm21cm/model/rhizome.py",
                    "ebm21cm/model/recurrent.py", "ebm21cm/data/cache.py", "ebm21cm/data/toy.py",
                    "ebm21cm/data/memory.py")
    hashes = {p: _sha256(root / p) for p in source_paths}
    eval_batch = config.eval_batch_size or config.batch_size
    channels = parse_channels(config.channels, json.loads(datasets["train"].attrs["bands"]))
    subset = train_cone_subset(cones["train"], config.train_cones)
    if config.data == "memory":
        load_start = time.perf_counter()
        storage = device if config.storage == "device" else torch.device("cpu")
        memory = {s: MemorySplit(cache_path, s, stats, channels, cones=subset if s == "train" else None,
                                 storage=storage, device=device) for s in ("train", "validation")}
        print(f"Loaded train/validation into {storage} memory: "
              f"{sum(m.nbytes for m in memory.values()) / 2**30:.1f} GiB, "
              f"{len(memory['train'])}/{len(memory['validation'])} rows, "
              f"{time.perf_counter() - load_start:.0f} s", flush=True)
        augmented = memory["train"]
        stream = MemoryStream(augmented, config.batch_size, config.seed)
        val_loader = memory["validation"].loader(eval_batch)
        train_rows = augmented.rows
    else:
        augmented = SliceDataset(cache_path, "train", stats, augment_data=True, seed=config.seed)
        stream = TrainingStream(augmented, config.batch_size, config.seed)
        val_loader = DataLoader(datasets["validation"], batch_size=eval_batch)
        train_rows = datasets["train"].rows
    model_config = {
        "in_channels": len(channels), "out_channels": 1,
        "scalar_dim": datasets["train"].n_scalars, "width": config.width, "modes": config.modes,
        "n_steps": config.updates, "step_size": config.step_size,
    }
    model = RhizomeOperator2d(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    probe_indices = np.random.default_rng(config.seed).choice(
        len(train_rows), min(config.train_probe_rows, len(train_rows)), replace=False)
    if config.data == "memory":
        probe_loader = augmented.loader(eval_batch, np.sort(probe_indices))
    else:
        probe_loader = DataLoader(Subset(datasets["train"], probe_indices.tolist()), batch_size=eval_batch)
    signature = {"config": {k: v for k, v in asdict(config).items() if k != "max_steps"},
                 "cache_sha256": cache_hash, "source_sha256": hashes,
                 "torch": str(torch.__version__), "device": str(device)}
    if initial_metadata is not None:
        if any(initial_metadata[k] != v for k, v in signature.items()):
            raise ValueError("incomplete initial setup differs from requested run")
        if config.max_steps < initial_metadata["initial_max_steps"]:
            raise ValueError("max_steps may only increase on resume")
        print("Recovering incomplete initial setup; restarting from the original seed.", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    step, samples_seen, elapsed = 0, 0, 0.0
    history, stopped = [], False
    window_loss, window_pixels = 0.0, 0
    if resume:
        checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["signature"] != signature:
            raise ValueError("resume config, cache, source, PyTorch version, or device differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        stream.load_state_dict(checkpoint["stream"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        step, samples_seen, elapsed = checkpoint["step"], checkpoint["samples_seen"], checkpoint["elapsed_seconds"]
        plateau = PlateauState(**checkpoint["plateau"])
        best_model = checkpoint["best_model"]
        history, stopped = checkpoint["history"], checkpoint["stopped"]
        window_loss, window_pixels = checkpoint["window_loss"], checkpoint["window_pixels"]
        if config.max_steps < checkpoint["max_steps_requested"]:
            raise ValueError("max_steps may only increase on resume")
        print(f"Resumed step {step}; best validation BCE {plateau.best:.6f}", flush=True)
    else:
        scores = validation_scores(model, val_loader, device)
        initial = scores["bce"]
        if not math.isfinite(initial):
            raise FloatingPointError("non-finite initial validation loss")
        plateau = PlateauState(initial, initial)
        best_model = _cpu_state(model)
        history.append({"step": 0, "train_window_bce": None,
                        "train_probe_bce": validation_bce(model, probe_loader, device),
                        "validation_bce": initial, "validation_rmse": scores["rmse"],
                        "validation_mixed_slice_rmse": scores["mixed_slice_rmse"],
                        "lr": config.lr, "next_lr": config.lr})
        metadata = {
            **signature, "model_config": model_config, "stats": stats,
            "cache": str(Path(cache_path).resolve()), "git_commit": git_commit(root),
            "splits": {s: {"rows": len(ds), "cones": sorted(cones[s])} for s, ds in datasets.items()},
            "channels": channels, "bands": [json.loads(datasets["train"].attrs["bands"])[j] for j in channels],
            "train_subset": None if subset is None else {"rows": len(train_rows), "cones": subset.tolist()},
            "train_probe_cache_rows": train_rows[probe_indices].tolist(),
            "architecture": "rhizome", "initial_max_steps": config.max_steps,
            "stopping_rule": "Validation plateau at minimum LR; not mathematical convergence.",
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    start = time.perf_counter()

    def save():
        _sync(device)
        saved_elapsed = elapsed + time.perf_counter() - start
        _atomic_save({
            "signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "model_config": model_config, "architecture": "rhizome", "stats": stats, "channels": channels,
            "stream": stream.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "step": step, "samples_seen": samples_seen, "elapsed_seconds": saved_elapsed,
            "max_steps_requested": config.max_steps,
            "plateau": asdict(plateau), "best_model": best_model, "history": history, "stopped": stopped,
            "window_loss": window_loss, "window_pixels": window_pixels,
        }, run_dir / "last.pt")
        # Recreate best.pt from the authoritative last checkpoint on resume,
        # including recovery from interruption between these two file writes.
        _atomic_save({"model": best_model, "model_config": model_config, "architecture": "rhizome",
                      "stats": stats, "channels": channels, "step": plateau.best_step,
                      "validation_bce": plateau.best},
                     run_dir / "best.pt")
        (run_dir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
        return saved_elapsed

    save()
    try:
        while step < config.max_steps and not stopped:
            model.train()
            batch = stream.next()
            cond, scalars, truth = _batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(cond, scalars), truth)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {step + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            samples_seen += truth.shape[0]
            window_loss += float(loss.detach()) * truth.numel()
            window_pixels += truth.numel()
            # Budget stops between checks preserve the partial loss window and
            # do not add a scheduler event absent from uninterrupted training.
            if step % config.val_every == 0:
                scores = validation_scores(model, val_loader, device)
                val = scores["bce"]
                lr = optimizer.param_groups[0]["lr"]
                next_lr, stopped, improved = plateau.observe(val, step, lr, config)
                if improved:
                    best_model = _cpu_state(model)
                for group in optimizer.param_groups:
                    group["lr"] = next_lr
                record = {
                    "step": step, "epochs_seen": samples_seen / len(augmented),
                    "train_window_bce": window_loss / window_pixels,
                    "train_probe_bce": validation_bce(model, probe_loader, device),
                    "validation_bce": val, "validation_rmse": scores["rmse"],
                    "validation_mixed_slice_rmse": scores["mixed_slice_rmse"],
                    "best_validation_bce": plateau.best,
                    "best_step": plateau.best_step, "lr": lr, "next_lr": next_lr,
                    "bad_checks": plateau.bad_checks, "lr_reductions": plateau.reductions,
                    "grad_norm": float(grad_norm),
                }
                history.append(record)
                window_loss, window_pixels = 0.0, 0
                save()
                print(f"step {step:6d} epochs {record['epochs_seen']:6.1f} "
                      f"train {record['train_window_bce']:.5f} val {val:.5f} "
                      f"mixed-rmse {scores['mixed_slice_rmse'] or float('nan'):.5f} "
                      f"best {plateau.best:.5f} lr {lr:.2g}->{next_lr:.2g} "
                      f"plateau {plateau.bad_checks}/{config.stop_patience}", flush=True)
    except KeyboardInterrupt:
        # last.pt already contains a consistent completed validation boundary.
        print("Interrupted: resume from the last completed checkpoint.", flush=True)
        raise
    training_seconds = save()
    _curves(run_dir / "learning_curves.png", history)
    result = {
        "stop_reason": "validation_plateau_at_min_lr" if stopped else "max_steps",
        "criterion_met": stopped, "steps": step, "samples_seen": samples_seen,
        "epochs_seen": samples_seen / len(augmented), "best_step": plateau.best_step,
        "best_validation_bce": plateau.best, "initial_validation_bce": history[0]["validation_bce"],
        "final_lr": optimizer.param_groups[0]["lr"], "lr_reductions": plateau.reductions,
        "elapsed_training_seconds": training_seconds,
        "max_steps_requested": config.max_steps, "plateau": asdict(plateau),
        "test_evaluated": False,
    }
    if stopped:
        best, _ = load_checkpoint(run_dir / "best.pt", device)
        if config.data == "memory":
            test_loader = MemorySplit(cache_path, "test", stats, channels, storage=storage,
                                      device=device).loader(eval_batch)
        else:
            test_loader = DataLoader(datasets["test"], batch_size=eval_batch)
        fields, _ = predict(best, test_loader, device)
        result["test"] = field_metrics(fields["prediction"], fields["truth"])
        result["test_evaluated"] = True
        np.savez_compressed(run_dir / "test_predictions.npz", **fields)
        rows = _example_rows(fields["truth"])
        density = np.stack([datasets["test"].raw(int(i))["delta"][datasets["test"].center_band] for i in rows])
        _plot_examples(run_dir / "examples.png",
                       {"rhizome": {k: fields[k][rows] for k in ("prediction", "truth", "z")}}, density)
    (run_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    defaults = asdict(LongRunConfig())
    for key, value in defaults.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args(argv)
    config = LongRunConfig(**{key: getattr(args, key) for key in defaults})
    run(args.cache, args.run_dir, config, args.device, args.resume)


if __name__ == "__main__":
    main()
