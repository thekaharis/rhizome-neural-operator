"""Run-directory helpers shared by training, sampling and evaluation."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import torch

from .data.cache import Normalizer
from .model import build_model

DEFAULTS = {
    "model": {"ch": 64, "mults": [1, 2, 4], "num_res": 2, "attn_levels": [2], "heads": 4,
              "dropout": 0.0},
    "train": {
        "batch_size": 16, "lr": 2e-4, "weight_decay": 0.0, "warmup": 1000, "steps": 300_000,
        "ema_decay": 0.9995, "grad_clip": 1.0, "sigma_data": None,
        # log(sigma) ~ N(p_mean, p_std). EDM's image defaults (-1.2, 1.2) almost never
        # train sigma > 3, where a conditional sampler fixes the global ionization state;
        # (-0.4, 1.4) covers sigma ~ 0.04-11 at +-2 std. Sampling starts at
        # sample_sigma_max, inside that range.
        "p_mean": -0.4, "p_std": 1.4, "sample_sigma_max": 10.0,
        "augment": True, "amp": "none", "workers": 4, "seed": 0, "log_every": 100,
        "val_every": 2000, "val_rows": 64,
        "val_sigmas": [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        "ckpt_every": 5000, "sample_every": 10_000, "sample_rows": 8, "sample_steps": 32,
    },
}


def merge_config(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(cfg: dict, pairs) -> dict:
    """``["train.lr=1e-4", "model.mults=[1,2]"]`` -> nested update (values parsed as JSON)."""
    cfg = copy.deepcopy(cfg)
    for p in pairs or []:
        key, _, raw = p.partition("=")
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        node = cfg
        *head, last = key.split(".")
        for h in head:
            node = node.setdefault(h, {})
        node[last] = val
    return cfg


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_run(run_dir, which="best", device="cpu", use_ema=True):
    """(model in eval mode, metadata dict, Normalizer) for a finished or running run."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    ck = torch.load(run_dir / f"{which}.pt", map_location="cpu", weights_only=False)
    model = build_model(meta["config"]["model"], meta["n_cond"], meta["n_scalars"], meta["sigma_data"])
    model.load_state_dict(ck["ema" if use_ema else "model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    meta["checkpoint_step"] = ck.get("step")
    return model, meta, Normalizer(meta["stats"])
