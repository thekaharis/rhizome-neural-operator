"""Train the energy-parameterized diffusion model on a slice cache.

    python -m ebm21cm.train --cache slices.h5 --run-dir runs/base \\
        [--config configs/base.json] [--set train.lr=1e-4 ...] [--resume]

Writes into the run directory:
    run_metadata.json   config, stats, cache geometry, versions
    metrics.jsonl       training log lines and validation records
    last.pt, best.pt    weights (+EMA, optimizer) ; best = lowest validation loss (EMA)
    samples/step*.png   periodic validation samples
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.cache import SliceDataset, read_stats
from .model import build_model
from .model.energy import edm_loss
from .run import DEFAULTS, apply_overrides, git_commit, merge_config, pick_device
from .sampling import sample
from . import metrics as M
from . import viz


def _log(fh, rec):
    fh.write(json.dumps(rec) + "\n")
    fh.flush()


def _batches(ds, batch_size, workers, seed, shuffle=True):
    # One loader for the whole run: workers persist across epochs (spawning them
    # is slow on macOS) and the seeded generator reshuffles every epoch.
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=workers,
                    generator=g, persistent_workers=workers > 0)
    while True:
        yield from dl


def _stack(ds, rows, device):
    items = [ds[j] for j in rows]
    return {k: torch.stack([it[k] for it in items]).to(device) for k in ("cond", "target", "scalars")}, items


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm.detach(), 1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def validate(model, batch, sigmas, sigma_data, seed=1234):
    """Deterministic denoising loss on fixed rows, per noise level."""
    out = {}
    tot = 0.0
    for j, s in enumerate(sigmas):
        g = torch.Generator().manual_seed(seed + j)
        x = batch["target"]
        noise = torch.randn(x.shape, generator=g).to(x.device)
        sig = torch.full((x.shape[0],), float(s), device=x.device)
        d, e = model.denoise(x + noise * s, sig, batch["cond"], batch["scalars"])
        w = (s ** 2 + sigma_data ** 2) / (s * sigma_data) ** 2
        mse = (d - x).pow(2).mean(dim=(0, 2, 3))
        loss = float(w * mse.mean())
        out[f"{s:g}"] = {"loss": loss, "mse_xhi": float(mse[0]), "mse_tb": float(mse[1]),
                         "energy_per_px": float(e.mean() / x[0].numel())}
        tot += loss
    return tot / len(sigmas), out


def sample_report(model, ds, rows, norm, cell, steps, path, sigma_max, seed=0):
    batch, items = _stack(ds, rows, next(model.parameters()).device)
    g = torch.Generator().manual_seed(seed)
    x, _ = sample(model, batch["cond"], batch["scalars"], n_steps=steps, sigma_max=sigma_max, generator=g)
    xhi_s, tb_s = (a.cpu().numpy() for a in norm.to_physical(x))
    xhi_t, tb_t = (a.cpu().numpy() for a in norm.to_physical(batch["target"]))
    rec = {"rmse_xhi": M.rmse(xhi_s, xhi_t), "rmse_tb": M.rmse(tb_s, tb_t),
           "hedge_xhi_sample": M.hedging_fraction(xhi_s), "hedge_xhi_truth": M.hedging_fraction(xhi_t)}
    center = ds.center_band
    delta = batch["cond"][:, center].cpu().numpy()
    viz.examples(path, delta, xhi_t, xhi_s[:, None], tb_t, tb_s[:, None],
                 [it["z"] for it in items], n_show_samples=1)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ebm21cm.train", description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", help="JSON with 'model' / 'train' sections, merged over defaults")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="overrides, e.g. train.lr=1e-4 model.ch=32")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true", help="continue from run-dir/last.pt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    run = Path(a.run_dir)
    cfg = merge_config(DEFAULTS, json.loads(Path(a.config).read_text()) if a.config else {})
    cfg = apply_overrides(cfg, a.set)
    if run.exists() and any(run.iterdir()) and not a.resume:
        raise SystemExit(f"{run} is not empty; pass --resume or choose a new --run-dir")
    run.mkdir(parents=True, exist_ok=True)
    (run / "samples").mkdir(exist_ok=True)
    tc = cfg["train"]
    device = pick_device(a.device)
    torch.manual_seed(tc["seed"])
    np.random.seed(tc["seed"])

    stats = read_stats(a.cache)
    train_ds = SliceDataset(a.cache, "train", stats, augment_data=tc["augment"], seed=tc["seed"])
    val_ds = SliceDataset(a.cache, "validation", stats)
    if len(val_ds) == 0:
        raise SystemExit("validation split is empty")
    sigma_data = float(tc["sigma_data"] or stats["sigma_data"])
    model = build_model(cfg["model"], train_ds.n_cond, train_ds.n_scalars, sigma_data).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    n_params = sum(p.numel() for p in model.parameters())

    meta_path = run / "run_metadata.json"
    if a.resume and meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old["config"]["model"] != cfg["model"]:
            raise SystemExit("model config differs from the run being resumed")
    meta = {
        "config": cfg, "stats": stats, "sigma_data": sigma_data, "cache": str(Path(a.cache).resolve()),
        "cache_attrs": dict(train_ds.attrs), "n_cond": train_ds.n_cond, "n_scalars": train_ds.n_scalars,
        "n_train_rows": len(train_ds), "n_val_rows": len(val_ds), "n_params": n_params,
        "torch": torch.__version__, "device": str(device),
        "git_commit": git_commit(Path(__file__).resolve().parent.parent),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    step, best = 0, math.inf
    if a.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
        print(f"resumed at step {step}")

    rng = np.random.default_rng(tc["seed"])
    val_rows = np.sort(rng.choice(len(val_ds), size=min(tc["val_rows"], len(val_ds)), replace=False))
    val_batch, _ = _stack(val_ds, val_rows, device)
    # Stratified in z so the periodic sample figure spans the history.
    order = np.argsort(val_ds.z)
    sample_rows = order[np.linspace(0, len(order) - 1, min(tc["sample_rows"], len(order))).astype(int)]
    cell = float(train_ds.attrs["cell_size_mpc"])

    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(tc["amp"])
    print(f"{n_params / 1e6:.2f}M parameters, {len(train_ds)} train / {len(val_ds)} val rows, "
          f"sigma_data={sigma_data:.3f}, device={device}")

    def save(name):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "step": step, "best": best, "config": cfg}, run / f"{name}.pt.tmp")
        (run / f"{name}.pt.tmp").replace(run / f"{name}.pt")

    log = open(run / "metrics.jsonl", "a")
    train_ds.seed = tc["seed"] + step  # a resumed run continues with fresh augmentations
    it = _batches(train_ds, tc["batch_size"], tc["workers"], tc["seed"] + step)
    t0, seen = time.time(), 0
    model.train()
    while step < tc["steps"]:
        b = next(it)
        cond, target, scal = (b[k].to(device, non_blocking=True) for k in ("cond", "target", "scalars"))
        lr = tc["lr"] * min(1.0, (step + 1) / max(tc["warmup"], 1))
        for gp in opt.param_groups:
            gp["lr"] = lr
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            loss, sig, _ = edm_loss(model, target, cond, scal, tc["p_mean"], tc["p_std"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        opt.step()
        # Warm-up: early EMA would otherwise stay anchored to the random init.
        ema_update(ema, model, min(tc["ema_decay"], (1 + step) / (10 + step)))
        step += 1
        seen += target.shape[0]

        if step % tc["log_every"] == 0:
            dt = time.time() - t0
            _log(log, {"step": step, "loss": float(loss.detach()), "lr": lr, "grad_norm": float(gn),
                       "samples_per_s": seen / dt})
            print(f"step {step} loss {float(loss.detach()):.4f} gn {float(gn):.2f} {seen / dt:.1f} samples/s")
            t0, seen = time.time(), 0
        if step % tc["val_every"] == 0 or step == tc["steps"]:
            vloss, per = validate(ema, val_batch, tc["val_sigmas"], sigma_data)
            is_best = vloss < best
            if is_best:
                best = vloss
                save("best")
            _log(log, {"step": step, "val_loss": vloss, "best": is_best, "per_sigma": per})
            print(f"step {step} val {vloss:.4f}{' (best)' if is_best else ''}")
        if step % tc["sample_every"] == 0 or step == tc["steps"]:
            rec = sample_report(ema, val_ds, sample_rows, val_ds.norm, cell, tc["sample_steps"],
                                run / "samples" / f"step{step:07d}.png", tc["sample_sigma_max"])
            _log(log, {"step": step, "val_samples": rec})
            print(f"step {step} samples {rec}")
        if step % tc["ckpt_every"] == 0 or step == tc["steps"]:
            save("last")
    log.close()
    print(f"done: {step} steps, best val {best:.4f}")


if __name__ == "__main__":
    main()
