"""Fine-tune the 21cmFAST slice-wise rhizome on LoReLi II.

The spectral kernels (transport) stay frozen in the first three ladder arms;
the pointwise ends adapt to LoReLi's continuous ionization field and its
different source/emissivity mapping.  Training starts from the reference
checkpoint with the 24 -> 48 mode remap, or from random init for the scratch
control.

    python -m ebm21cm.train_loreli_finetune \
        --cache runs/loreli_cache/loreli_slices_l294.h5 \
        --run-dir runs/loreli_finetune/l294_b8_decoder_seed0 \
        --init-from runs/rhizome_mf2d_w64_structured_all \
        --freeze decoder --train-blocks 8 --seed 0 --device cuda

The loss is BCE on the continuous x_HI target plus the reference normalized
T_b MSE, with both terms restricted to the valid 206x206 interior of the
280-cell domain.  Selection is the mean normalized MSE on the validation
blocks; the learning rate is reduced on a plateau and training stops at the
minimum rate.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F

from .data.loreli_cache import SCHEMA
from .data.loreli_memory import LoReLiMemorySplit, LoReLiStream
from .data.loreli_splits import load_split, nested_train_blocks
from .model.freeze import (
    FREEZE_CHOICES,
    freeze_model,
    optimizer_parameter_groups,
    parameter_groups,
)
from .model.remap import remapped_model
from .model.rhizome import build_model
from .run import git_commit
from .train_rhizome import PlateauState, _atomic_save, _cpu_state, _sha256
from .train_rhizome_mf2d import tb_normalized


@dataclass
class FinetuneConfig:
    max_steps: int = 6000
    min_steps: int = 1500
    batch_size: int = 8
    eval_batch_size: int = 8
    lr_new: float = 1e-3
    lr_pretrained: float = 1e-4
    min_lr: float = 1e-5
    lr_factor: float = 0.5
    lr_patience: int = 4
    stop_patience: int = 8
    relative_min_delta: float = 0.002
    val_every: int = 200
    seed: int = 0
    threads: int = 4
    tb_weight: float = 1.0
    train_blocks: int = 0
    storage: str = "cpu"

    def validate(self) -> None:
        for name in ("max_steps", "min_steps", "batch_size", "eval_batch_size",
                     "lr_patience", "stop_patience", "val_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_steps > self.max_steps:
            raise ValueError("min_steps cannot exceed max_steps")
        for name in ("lr_new", "lr_pretrained", "min_lr"):
            value = getattr(self, name)
            if not (value >= 0 and math.isfinite(value)):
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.min_lr > min(self.lr_new, self.lr_pretrained):
            raise ValueError("min_lr must not exceed either learning rate")
        if not 0 < self.lr_factor < 1:
            raise ValueError("lr_factor must be in (0, 1)")
        if not 0 <= self.relative_min_delta < 1:
            raise ValueError("relative_min_delta must be in [0, 1)")
        if self.storage not in ("cpu", "device"):
            raise ValueError("storage must be 'cpu' or 'device'")


def load_reference(run: str | Path) -> tuple[torch.nn.Module, dict, dict]:
    """The LoReLi-domain model for a reference checkpoint.

    Returns ``(model, model_config, checkpoint)``.  Reference checkpoints are
    trained at 24 modes on the 200 Mpc box and remapped to 48 modes; a
    checkpoint already saved on the LoReLi domain (``domain_factor=2``) is
    loaded directly, without a second remap.
    """
    import torch

    run = Path(run)
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    if best.get("node_phase"):
        raise SystemExit(f"{run}: node-phase checkpoint; LoReLi has no node interpolation")
    config = dict(best["model_config"])
    if config.get("in_channels") != 26:
        raise SystemExit(f"{run}: expected 26 conditioning channels, got {config.get('in_channels')}")
    if config.get("out_channels") != 2:
        raise SystemExit(f"{run}: expected two output channels, got {config.get('out_channels')}")
    if best.get("head") != "structured":
        raise SystemExit(f"{run}: expected the structured T_b head")
    already = int(best.get("domain_factor", 1)) > 1
    if already:
        model_config = dict(config)
        model = build_model(model_config)
        model.load_state_dict(best["model"])
    else:
        model_config = dict(config)
        model_config["modes"] = int(model_config["modes"]) * 2
        if "coarse_modes" in model_config:
            model_config["coarse_modes"] = int(model_config["coarse_modes"]) * 2
        model = remapped_model(config, best["model"], factor=2)
    return model, model_config, best


def build_finetune_model(init_from: str | Path, *, scratch: bool = False):
    """Model at 48 modes, either the remapped reference or random init."""
    model, config, best = load_reference(init_from)
    if scratch:
        model = build_model(config)
        model_config = config
    else:
        model_config = config
    return model, model_config, best


def _batch_loss(model, batch, tb_norm, tb_weight, *, head="structured"):
    out = model(batch["cond"], batch["scalars"])
    target = batch["target"]
    mask = batch["mask"].to(dtype=torch.float32)
    bce = F.binary_cross_entropy_with_logits(
        out[:, :1], target[:, :1], reduction="none"
    )
    tb_pred = tb_normalized(out, batch["phys"], head, *tb_norm)
    tb_mse = (tb_pred - target[:, 1:2]).square()
    denominator = mask.sum().clamp_min(1.0)
    loss = (bce * mask).sum() / denominator
    loss = loss + tb_weight * (tb_mse * mask).sum() / denominator
    return loss, out, bce, tb_mse, mask, target


@torch.no_grad()
def evaluate(model, loader, device, tb_norm, tb_weight, head="structured"):
    model.eval()
    totals = {"n": 0.0, "bce": 0.0, "x_mse": 0.0, "tb_mse": 0.0, "loss": 0.0}
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        loss, out, bce, tb_mse, mask, target = _batch_loss(
            model, batch, tb_norm, tb_weight, head=head
        )
        count = float(mask.sum())
        totals["n"] += count
        totals["loss"] += float(loss) * count
        totals["bce"] += float((bce * mask).sum())
        totals["x_mse"] += float(((out[:, :1].sigmoid() - target[:, :1]).square() * mask).sum())
        totals["tb_mse"] += float((tb_mse * mask).sum())
    if totals["n"] == 0:
        raise ValueError("empty validation loader")
    x_mse = totals["x_mse"] / totals["n"]
    tb_mse = totals["tb_mse"] / totals["n"]
    return {
        "loss": totals["loss"] / totals["n"],
        "bce": totals["bce"] / totals["n"],
        "xhi_rmse": math.sqrt(x_mse),
        "xhi_nmse": x_mse,
        "tb_nmse": tb_mse,
        "tb_rmse_mk": math.sqrt(tb_mse) * float(tb_norm[1]),
        "score": 0.5 * (x_mse + tb_mse),
    }


def _signature(cache, init_from, config, freeze, train_blocks, split, source_root):
    return {
        "cache_sha256": _sha256(cache),
        "init_from": str(Path(init_from).resolve()),
        "checkpoint_sha256": _sha256(Path(init_from) / "best.pt"),
        "freeze": freeze,
        "train_blocks": list(train_blocks),
        "split_sha256": split["sha256"],
        "config": {k: v for k, v in asdict(config).items() if k != "max_steps"},
        "device": config.storage,
        "torch": str(torch.__version__),
        "source_sha256": {
            path: _sha256(source_root / path)
            for path in (
                "ebm21cm/train_loreli_finetune.py",
                "ebm21cm/data/loreli_cache.py",
                "ebm21cm/data/loreli_memory.py",
                "ebm21cm/data/loreli_splits.py",
                "ebm21cm/model/freeze.py",
                "ebm21cm/model/remap.py",
                "ebm21cm/model/rhizome.py",
                "ebm21cm/train_rhizome_mf2d.py",
            )
        },
    }


def run(
    cache: str | Path,
    run_dir: str | Path,
    init_from: str | Path,
    *,
    freeze: str = "decoder",
    scratch: bool = False,
    split_file: str | Path,
    config: FinetuneConfig | None = None,
    device: str = "cuda",
    resume: bool = False,
    log=print,
) -> dict:
    config = config or FinetuneConfig()
    config.validate()
    if freeze not in FREEZE_CHOICES:
        raise ValueError(f"freeze must be one of {FREEZE_CHOICES}")
    if scratch and freeze != "none":
        raise ValueError("--scratch trains every parameter; use --freeze none")
    device = torch.device(device)
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    run_dir = Path(run_dir)
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("run directory is not empty; use --resume or a new directory")

    with h5py.File(cache, "r") as h:
        if h.attrs.get("schema") != SCHEMA:
            raise ValueError(f"{cache}: not an {SCHEMA} cache")
        cache_attrs = dict(h.attrs)
    split = load_split(split_file)
    if cache_attrs.get("split_sha256") != split["sha256"]:
        raise ValueError(
            "the cache was built from a different block-split file; rebuild it or use the matching split"
        )
    train_blocks = nested_train_blocks(split, config.train_blocks)
    if not train_blocks:
        raise ValueError("no training blocks selected")
    model, model_config, best = build_finetune_model(init_from, scratch=scratch)
    tb_norm = tuple(float(v) for v in best["tb_norm"])
    scalar_keep = best.get("scalar_keep")
    model = model.to(device)

    report = freeze_model(model, freeze)
    groups = parameter_groups(model)
    optim_groups = optimizer_parameter_groups(model, scratch=scratch)
    trainable = report["trainable"]
    if trainable == 0:
        raise ValueError("the selected freeze setting leaves no trainable parameters")
    optimizers = []
    if optim_groups["new"]:
        optimizers.append({"params": optim_groups["new"], "lr": config.lr_new, "name": "new"})
    if optim_groups["pretrained"]:
        optimizers.append({"params": optim_groups["pretrained"], "lr": config.lr_pretrained,
                           "name": "pretrained"})
    optimizer = torch.optim.Adam(optimizers)
    log(f"freeze={freeze}: trainable {trainable:,} / {report['total']:,} parameters "
        f"({100 * trainable / report['total']:.2f}%)")
    for name in ("decoder", "ends", "pointwise", "spectral"):
        log(f"  {name:9s} {report['trainable_counts'][name]:>12,} / {report['counts'][name]:>12,}")

    storage = device if config.storage == "device" else torch.device("cpu")
    t0 = time.perf_counter()
    train = LoReLiMemorySplit(cache, "train", blocks=train_blocks, storage=storage, device=device)
    val = LoReLiMemorySplit(cache, "val", blocks=split["val_blocks"], storage=storage, device=device)
    if scalar_keep is not None:
        keep = torch.as_tensor(scalar_keep, dtype=torch.long)
        train.scalars = train.scalars[:, keep]
        val.scalars = val.scalars[:, keep]
    log(f"loaded {len(train)} train / {len(val)} val rows ({train.nbytes / 2**30:.1f} GiB) "
        f"in {time.perf_counter() - t0:.0f} s")
    if not len(train) or not len(val):
        raise ValueError("empty train or validation split")

    root = Path(__file__).resolve().parents[1]
    signature = _signature(cache, init_from, config, freeze, train_blocks, split, root)
    stream = LoReLiStream(train, config.batch_size, config.seed)
    val_loader = val.loader(config.eval_batch_size)

    step, samples, elapsed, history, stopped = 0, 0, 0.0, [], False
    if resume:
        checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["signature"] != signature:
            raise ValueError("resume cache, init checkpoint, split, freeze or config differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        stream.load_state_dict(checkpoint["stream"])
        torch.set_rng_state(checkpoint["torch_rng"])
        step, samples, elapsed = checkpoint["step"], checkpoint["samples_seen"], checkpoint["elapsed_seconds"]
        plateau = PlateauState(**checkpoint["plateau"])
        best_model, history, stopped = checkpoint["best_model"], checkpoint["history"], checkpoint["stopped"]
        log(f"resumed at step {step}; best validation score {plateau.best:.6f}")
    else:
        scores = evaluate(model, val_loader, device, tb_norm, config.tb_weight)
        plateau = PlateauState(scores["score"], scores["score"])
        best_model = _cpu_state(model)
        history.append({"step": 0, **{f"val_{k}": v for k, v in scores.items()},
                        "lr_new": config.lr_new, "lr_pretrained": config.lr_pretrained})
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_metadata.json").write_text(json.dumps({
            **signature,
            "model_config": model_config,
            "cache": str(Path(cache).resolve()),
            "cache_attrs": {k: (v.item() if hasattr(v, "item") else v) for k, v in cache_attrs.items()},
            "freeze_counts": report["counts"],
            "trainable_counts": report["trainable_counts"],
            "trainable_total": trainable,
            "parameter_groups": {k: len(v) for k, v in groups.items()},
            "train_blocks": train_blocks,
            "train_rows": len(train),
            "val_rows": len(val),
            "split": {k: split[k] for k in ("train_blocks", "val_blocks", "test_blocks", "sha256")},
            "objective": "masked BCE(x_HI) + tb_weight * MSE(normalized T_b); selection: mean normalized MSE",
            "git_commit": git_commit(root),
        }, indent=2))
    start = time.perf_counter()

    def save():
        saved = elapsed + time.perf_counter() - start
        _atomic_save({
            "signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "stream": stream.state_dict(), "torch_rng": torch.get_rng_state(),
            "step": step, "samples_seen": samples, "elapsed_seconds": saved,
            "plateau": asdict(plateau), "best_model": best_model, "history": history,
            "stopped": stopped,
        }, run_dir / "last.pt")
        _atomic_save({
            "model": best_model, "model_config": model_config, "head": best.get("head", "structured"),
            "tb_norm": tb_norm, "step": plateau.best_step, "validation_score": plateau.best,
            "domain_factor": 2, "freeze": freeze, "train_blocks": train_blocks,
            "init_from": str(Path(init_from).resolve()), "scratch": bool(scratch),
            "cache": str(Path(cache).resolve()),
            **({"scalar_keep": scalar_keep} if scalar_keep is not None else {}),
        }, run_dir / "best.pt")
        (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
        return saved

    save()
    current_lr_new = next(
        (float(group["lr"]) for group in optimizer.param_groups if group.get("name") == "new"),
        config.lr_new,
    )
    while step < config.max_steps and not stopped:
        model.train()
        batch = stream.next()
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss, out, _, _, _, _ = _batch_loss(model, batch, tb_norm, config.tb_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step + 1}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            1.0, error_if_nonfinite=True,
        )
        optimizer.step()
        step += 1
        samples += int(batch["cond"].shape[0])
        if step % config.val_every == 0:
            scores = evaluate(model, val_loader, device, tb_norm, config.tb_weight)
            next_lr, stopped, improved = plateau.observe(scores["score"], step, current_lr_new, config)
            if improved:
                best_model = _cpu_state(model)
            ratio = next_lr / current_lr_new if current_lr_new else 1.0
            for group in optimizer.param_groups:
                group["lr"] *= ratio
            current_lr_new = next_lr
            history.append({
                "step": step, "epochs_seen": samples / len(train),
                "train_loss": float(loss.detach()), **{f"val_{k}": v for k, v in scores.items()},
                "best_score": plateau.best, "best_step": plateau.best_step,
                "lr_new": current_lr_new,
                "lr_pretrained": [g["lr"] for g in optimizer.param_groups if g.get("name") == "pretrained"],
                "bad_checks": plateau.bad_checks, "lr_reductions": plateau.reductions,
                "grad_norm": float(grad_norm),
            })
            save()
            log(f"step {step:6d} epochs {samples / len(train):5.2f} loss {float(loss.detach()):.5f} "
                f"val x_HI {scores['xhi_rmse']:.5f} T_b {scores['tb_rmse_mk']:.2f} mK "
                f"score {scores['score']:.6f} best {plateau.best:.6f} lr {current_lr_new:.2g} "
                f"plateau {plateau.bad_checks}/{config.stop_patience}")
    seconds = save()
    result = {
        "stop_reason": "validation_plateau_at_min_lr" if stopped else "max_steps",
        "steps": step, "samples_seen": samples, "epochs_seen": samples / len(train),
        "best_step": plateau.best_step, "best_validation_score": plateau.best,
        "elapsed_training_seconds": seconds, "freeze": freeze, "scratch": bool(scratch),
        "train_blocks": train_blocks, "trainable_parameters": trainable,
    }
    (run_dir / "results.json").write_text(json.dumps(result, indent=2))
    log(json.dumps(result, indent=2))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cache", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--init-from", required=True,
                   help="reference run directory (best.pt); also supplies the config for --scratch")
    p.add_argument("--split-file", default="runs/loreli_splits/block_split_seed0.json")
    p.add_argument("--freeze", choices=FREEZE_CHOICES, default="decoder")
    p.add_argument("--scratch", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    for field in fields(FinetuneConfig):
        p.add_argument("--" + field.name.replace("_", "-"), type=type(field.default),
                       default=field.default)
    args = p.parse_args(argv)
    config = FinetuneConfig(**{f.name: getattr(args, f.name) for f in fields(FinetuneConfig)})
    run(args.cache, args.run_dir, args.init_from, freeze=args.freeze, scratch=args.scratch,
        split_file=args.split_file, config=config, device=args.device, resume=args.resume,
        log=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
