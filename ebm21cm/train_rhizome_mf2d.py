"""Multi-field 2-D slice-wise rhizome: density + LOS-velocity bands -> x_HI + T_b.

    python -m ebm21cm.train_rhizome_mf2d --cache $WORK/data/compressed/rhizome_mf_slices.h5 \\
        --run-dir runs/rhizome_mf2d_w64 --device cuda --width 64 --batch-size 32 --head structured

The 2-D counterpart of ``train_rhizome3d`` multi-field run C, on an
``ebm21cm.data.mf_slices`` cache (fno-21cm's multi-field preparation, splits and
normalization, global-history emulators as scalar band means). Same schedule
and stopping rule as ``train_rhizome`` (Adam, LR reduced on a validation
plateau, stop at the floor), with the model selected and the plateau tracked on
fno-21cm's multi-field score: the mean over x_HI and T_b of the normalized MSE.

Output channel 0 is the x_HI logit. ``--head structured`` reads channel 1 as the
spin variable u of T_b = x_HI * phys * (1 - exp(u)), ``phys`` = A(z)(1+delta)V
from the cache (the 3-D ``StructuredBrightness3d`` physics on one slice);
``--head plain`` reads it as normalized T_b. Loss: BCE(x_HI) + tb_weight * MSE(T_b).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data.memory import MemoryStream
from .data.mf_slices import MFMemorySplit
from .model.rhizome import RhizomeOperator2d
from .run import git_commit
from .train_rhizome import LongRunConfig, PlateauState, _atomic_save, _cpu_state, _sha256

U_MAX = 5.0


@dataclass
class MFConfig(LongRunConfig):
    head: str = "structured"
    tb_weight: float = 1.0


def tb_normalized(out, phys, head, tb_offset, tb_scale):
    """Normalized T_b from the network output (B, 2, H, W)."""
    if head == "plain":
        return out[:, 1:2]
    tb = out[:, :1].sigmoid() * phys * (1.0 - torch.exp(out[:, 1:2].clamp(max=U_MAX)))
    return (tb - tb_offset) / tb_scale


@torch.no_grad()
def evaluate(model, loader, device, head, tb_norm, collect=False):
    model.eval()
    sums = {"bce": 0.0, "x_se": 0.0, "t_se": 0.0, "n": 0}
    kept = {"x_pred": [], "x_true": [], "t_pred": [], "t_true": [], "z": [], "cone_id": []}
    for batch in loader:
        out = model(batch["cond"], batch["scalars"])
        t = tb_normalized(out, batch["phys"], head, *tb_norm)
        x, y = out[:, :1].sigmoid(), batch["target"]
        sums["bce"] += float(F.binary_cross_entropy_with_logits(out[:, :1], y[:, :1], reduction="sum"))
        sums["x_se"] += float((x - y[:, :1]).square().sum())
        sums["t_se"] += float((t - y[:, 1:]).square().sum())
        sums["n"] += y[:, :1].numel()
        if collect:
            for k, v in (("x_pred", x), ("x_true", y[:, :1]), ("t_pred", t), ("t_true", y[:, 1:])):
                kept[k].append(v[:, 0].half().cpu().numpy())
            kept["z"].append(batch["z"].numpy())
            kept["cone_id"].append(batch["cone_id"].numpy())
    n = sums["n"]
    x_mse, t_nmse = sums["x_se"] / n, sums["t_se"] / n
    scores = {"bce": sums["bce"] / n, "xhi_rmse": math.sqrt(x_mse), "tb_rmse_mk": math.sqrt(t_nmse) * tb_norm[1],
              "xhi_nmse": x_mse, "tb_nmse": t_nmse, "score": 0.5 * (x_mse + t_nmse)}
    return (scores, {k: np.concatenate(v) for k, v in kept.items()}) if collect else scores


def run(cache_path, run_dir, config, device="cuda", resume=False):
    config.validate()
    if config.head not in ("structured", "plain"):
        raise ValueError("head must be structured or plain")
    device = torch.device(device)
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    run_dir = Path(run_dir)
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("run directory is not empty; use --resume or a new directory")
    storage = device if config.storage == "device" else torch.device("cpu")
    t0 = time.perf_counter()
    train = MFMemorySplit(cache_path, "train", storage, device)
    val = MFMemorySplit(cache_path, "val", storage, device)
    print(f"Loaded train/val: {(train.nbytes + val.nbytes) / 2**30:.1f} GiB, {len(train)}/{len(val)} rows, "
          f"{time.perf_counter() - t0:.0f} s", flush=True)
    norm = json.loads(train.attrs["normalization"])["brightness_temp"]
    tb_norm = (float(norm["offset"]), float(norm["scale"]))
    eval_batch = config.eval_batch_size or config.batch_size
    stream = MemoryStream(train, config.batch_size, config.seed)
    val_loader = val.loader(eval_batch)
    model_config = {"in_channels": train.cond.shape[1], "out_channels": 2, "scalar_dim": train.scalars.shape[1],
                    "width": config.width, "modes": config.modes, "n_steps": config.updates,
                    "step_size": config.step_size}
    model = RhizomeOperator2d(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    root = Path(__file__).resolve().parent.parent
    signature = {"config": {k: v for k, v in asdict(config).items() if k != "max_steps"},
                 "cache_sha256": _sha256(cache_path),
                 "source_sha256": {p: _sha256(root / p) for p in ("ebm21cm/train_rhizome_mf2d.py",
                                                                  "ebm21cm/data/mf_slices.py",
                                                                  "ebm21cm/model/rhizome.py")},
                 "torch": str(torch.__version__), "device": str(device)}
    step, samples, elapsed, history, stopped = 0, 0, 0.0, [], False
    window_loss, window_n = 0.0, 0
    if resume:
        ck = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        if ck["signature"] != signature:
            raise ValueError("resume config, cache, source, PyTorch version, or device differs")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        stream.load_state_dict(ck["stream"])
        torch.set_rng_state(ck["torch_rng"])
        step, samples, elapsed = ck["step"], ck["samples_seen"], ck["elapsed_seconds"]
        plateau, best_model, history, stopped = PlateauState(**ck["plateau"]), ck["best_model"], ck["history"], ck["stopped"]
        print(f"Resumed step {step}; best validation score {plateau.best:.6f}", flush=True)
    else:
        scores = evaluate(model, val_loader, device, config.head, tb_norm)
        plateau, best_model = PlateauState(scores["score"], scores["score"]), _cpu_state(model)
        history.append({"step": 0, **{f"val_{k}": v for k, v in scores.items()}, "lr": config.lr})
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metadata.json").write_text(json.dumps({
            **signature, "model_config": model_config, "cache": str(Path(cache_path).resolve()),
            "cache_attrs": {k: (v.item() if hasattr(v, "item") else v) for k, v in train.attrs.items()},
            "tb_normalization": tb_norm, "git_commit": git_commit(root),
            "rows": {"train": len(train), "val": len(val)},
            "objective": "BCE(x_HI) + tb_weight * MSE(normalized T_b); selection: mean normalized MSE",
            "initial_max_steps": config.max_steps}, indent=2))
    start = time.perf_counter()

    def save():
        saved = elapsed + time.perf_counter() - start
        _atomic_save({"signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "model_config": model_config, "stream": stream.state_dict(), "torch_rng": torch.get_rng_state(),
                      "step": step, "samples_seen": samples, "elapsed_seconds": saved, "plateau": asdict(plateau),
                      "best_model": best_model, "history": history, "stopped": stopped}, run_dir / "last.pt")
        _atomic_save({"model": best_model, "model_config": model_config, "head": config.head, "tb_norm": tb_norm,
                      "step": plateau.best_step, "validation_score": plateau.best}, run_dir / "best.pt")
        (run_dir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
        return saved

    save()
    while step < config.max_steps and not stopped:
        model.train()
        batch = stream.next()
        y = batch["target"]
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["cond"], batch["scalars"])
        loss = F.binary_cross_entropy_with_logits(out[:, :1], y[:, :1]) + config.tb_weight * F.mse_loss(
            tb_normalized(out, batch["phys"], config.head, *tb_norm), y[:, 1:])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step + 1}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        samples += y.shape[0]
        window_loss += float(loss.detach())
        window_n += 1
        if step % config.val_every == 0:
            scores = evaluate(model, val_loader, device, config.head, tb_norm)
            lr = optimizer.param_groups[0]["lr"]
            next_lr, stopped, improved = plateau.observe(scores["score"], step, lr, config)
            if improved:
                best_model = _cpu_state(model)
            for group in optimizer.param_groups:
                group["lr"] = next_lr
            history.append({"step": step, "epochs_seen": samples / len(train), "train_loss": window_loss / window_n,
                            **{f"val_{k}": v for k, v in scores.items()}, "best_score": plateau.best,
                            "best_step": plateau.best_step, "lr": lr, "next_lr": next_lr,
                            "bad_checks": plateau.bad_checks, "lr_reductions": plateau.reductions,
                            "grad_norm": float(grad_norm)})
            window_loss, window_n = 0.0, 0
            save()
            print(f"step {step:6d} epochs {samples / len(train):5.1f} loss {history[-1]['train_loss']:.5f} "
                  f"val x_HI {scores['xhi_rmse']:.5f} T_b {scores['tb_rmse_mk']:.3f} mK score {scores['score']:.6f} "
                  f"best {plateau.best:.6f} lr {lr:.2g}->{next_lr:.2g} plateau {plateau.bad_checks}/{config.stop_patience}",
                  flush=True)
    seconds = save()
    result = {"stop_reason": "validation_plateau_at_min_lr" if stopped else "max_steps", "steps": step,
              "samples_seen": samples, "epochs_seen": samples / len(train), "best_step": plateau.best_step,
              "best_validation_score": plateau.best, "elapsed_training_seconds": seconds}
    if stopped:
        model.load_state_dict(best_model)
        test = MFMemorySplit(cache_path, "test", storage, device)
        scores, kept = evaluate(model, test.loader(eval_batch), device, config.head, tb_norm, collect=True)
        result["test"] = scores
        np.savez_compressed(run_dir / "test_predictions.npz", **kept)
    (run_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cache", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    for f in fields(MFConfig):
        if f.name in ("data", "channels", "train_cones"):
            continue
        p.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    a = p.parse_args(argv)
    config = MFConfig(**{f.name: getattr(a, f.name) for f in fields(MFConfig)
                         if f.name not in ("data", "channels", "train_cones")})
    run(a.cache, a.run_dir, config, a.device, a.resume)


if __name__ == "__main__":
    main()
