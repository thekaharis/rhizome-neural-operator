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

Node phase (default off): w is the slice's 21cmFAST interpolation weight between its
two coeval node boxes, x = (1 - w) X_a + w X_b (``data.lightcone.node_weight``,
``--w-mode z|chi``, default z as the suite's cones confirm). ``--node-phase-input``
appends w to the scalars.
``--two-node-head`` adds output channel 2, d: the decoder's logits l_a (channel 0)
and l_b = l_a - softplus(d) (``--no-node-monotone``: l_b = d) give
x = (1 - w) sigmoid(l_a) + w sigmoid(l_b), collapsed in log space to the x_HI logit
log x - log(1 - x) that every consumer of channel 0 reads (``two_node_output``).
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

from .data.lightcone import W_MODES
from .data.memory import MemoryStream
from .data.mf_slices import MFMemorySplit
from .model.rhizome import build_model
from .run import git_commit
from .train_rhizome import LongRunConfig, PlateauState, _atomic_save, _cpu_state, _sha256

U_MAX = 5.0


@dataclass
class MFConfig(LongRunConfig):
    head: str = "structured"
    tb_weight: float = 1.0
    # Small-scale T_b power term: mean over slices and radial k bins (k >= power_kmin
    # Mpc^-1, power_bins log-spaced bins to the grid Nyquist) of
    # (log(P_pred + eps) - log(P_true + eps))^2 on mean-subtracted normalized T_b.
    power_weight: float = 0.0
    xhi_power_weight: float = 0.0  # the same term on the x_HI probability (sigmoid output)
    power_kmin: float = 0.3
    power_bins: int = 8
    power_eps: float = 1e-6
    # Compact real-space kernel of this radius in cells instead of the Fourier one (0 = spectral, --modes used).
    radius: float = 0.0
    rank: int = 0  # factorized compact kernel (0 = dense C x C per offset)
    # Ablation: drop the emulated global-history band means of x_HI ("xhi"), T_b ("tb") or both ("both")
    # from the scalars; the kept column indices are saved with the checkpoint for the exporter.
    drop_history: str = ""
    # Two-level RhizomeUNet2d when > 0: a coarse rollout on the grid average-pooled by this factor, upsampled into
    # the fine forcing. The fine level uses modes / radius / rank / updates as above.
    coarse_factor: int = 0
    coarse_modes: int = 0       # 0: every mode the coarse grid supports
    coarse_updates: int = 3
    coarse_radius: float = 0.0  # > 0: compact kernel of this radius in coarse cells
    coarse_rank: int = 0
    # Gate ablation (StateDependentIntegral2d): full | frozen | none | source | receiver | static.
    gate_mode: str = "full"
    # "lo:hi": every training step rolls out T ~ U{lo, ..., hi} tied updates (one T per batch, reproducible from
    # seed and step), so the decoder cannot tune itself to one rollout length; validation stays at --updates.
    random_updates: str = ""
    random_coarse_updates: str = ""  # the same for the coarse level of the two-level model ("" = fixed)
    # Multi-pathway interaction (MultiPathIntegral2d, single-level spectral model): R pathways, each with its own
    # gates, values and Fourier kernel of kernel_groups dense channel blocks (1 = dense C x C, width = depthwise),
    # concatenated into the proposal. share_pathways: "" | gates | kernels ties that part across pathways.
    pathways: int = 1
    kernel_groups: int = 1
    share_pathways: str = ""
    # The last disc_pathways pathways use a depthwise disc kernel of disc_radius cells applied by FFT (DiscFFTConv2d)
    # instead of the Fourier one: compact in space, every Fourier mode, cost independent of the radius.
    disc_radius: float = 0.0
    disc_pathways: int = 0
    disc_norm: str = ""  # "l1": disc weights g * v / sum|v| per channel, total weight independent of the disc area
    # Node phase w of each slice (see the module docstring): as a scalar input and/or through the two-node x_HI head.
    node_phase_input: bool = False
    two_node_head: bool = False
    node_monotone: bool = True  # two-node head: l_b = l_a - softplus(d), a cell does not recombine between nodes
    w_mode: str = "z"           # "z" (21cmFAST's interpolation, confirmed on the suite): linear in redshift


class PowerLoss(torch.nn.Module):
    """Per-slice log-ratio of binned transverse power above ``kmin`` (periodic 200 Mpc box)."""

    def __init__(self, h, w, kmin, n_bins, eps, box_mpc=200.0):
        super().__init__()
        ky = torch.fft.fftfreq(h, d=box_mpc / h) * 2 * math.pi
        kx = torch.fft.rfftfreq(w, d=box_mpc / w) * 2 * math.pi
        k = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
        edges = torch.logspace(math.log10(kmin), math.log10(float(k.max()) + 1e-6), n_bins + 1)
        index = torch.bucketize(k, edges) - 1  # -1 below kmin, n_bins beyond the last edge
        onehot = torch.stack([(index == b).float() for b in range(n_bins)])  # (n_bins, h, w//2+1)
        keep = onehot.sum((1, 2)) > 0
        self.register_buffer("weights", (onehot[keep] / onehot[keep].sum((1, 2), keepdim=True)).flatten(1))
        self.eps = eps

    def binned(self, field):
        f = field[:, 0].float()
        spec = torch.fft.rfft2(f - f.mean((-2, -1), keepdim=True), norm="ortho").abs().square()
        return spec.flatten(1) @ self.weights.T  # (B, n_bins)

    def per_slice(self, pred, true):
        return (torch.log(self.binned(pred) + self.eps) - torch.log(self.binned(true) + self.eps)).square().mean(1)

    def forward(self, pred, true):
        return self.per_slice(pred, true).mean()


HISTORY_PREFIX = {"xhi": ("x_HI_global",), "tb": ("T_b_global",), "both": ("x_HI_global", "T_b_global")}


def scalar_keep(names, drop):
    """Indices of the scalars kept after dropping the history band means named by ``drop``."""
    if not drop:
        return list(range(len(names)))
    if drop not in HISTORY_PREFIX:
        raise ValueError(f"drop_history must be one of {sorted(HISTORY_PREFIX)}")
    keep = [i for i, n in enumerate(names) if not n.startswith(HISTORY_PREFIX[drop])]
    if len(keep) == len(names):
        raise ValueError(f"no {drop} history scalars in the cache")
    return keep


def node_phase(config):
    """The checkpoint's node-phase settings, or None when neither node-phase option is on (the default)."""
    if config.w_mode not in W_MODES:
        raise ValueError(f"w_mode must be one of {W_MODES}")
    if not (config.node_phase_input or config.two_node_head):
        return None
    return {"w_mode": config.w_mode, "input": config.node_phase_input, "two_node": config.two_node_head,
            "monotone": config.node_monotone}


def node_scalars(scalars, w, node):
    """``scalars`` with the node phase w (B,) appended as the last column if the model takes it as input."""
    return torch.cat([scalars, w[:, None].to(scalars.dtype)], 1) if node and node["input"] else scalars


def two_node_output(out, w, monotone=True):
    """(B, 3, H, W) two-node output -> (B, 2, H, W) with channel 0 the x_HI logit log x - log(1 - x) of

        x = (1 - w) sigmoid(l_a) + w sigmoid(l_b),   l_a = out[:, 0],  l_b = l_a - softplus(out[:, 2]) or out[:, 2].

    log x and log(1 - x) are log-sum-exps of log-sigmoids, so the logit is finite for finite logits and any w in
    [0, 1] (no clamping needed), its sigmoid is x, and BCE-with-logits on it is the log-space BCE of x."""
    w = w.to(out.dtype).view(-1, 1, 1, 1)
    l_a = out[:, :1]
    l_b = l_a - F.softplus(out[:, 2:3]) if monotone else out[:, 2:3]
    log_a, log_b = torch.log1p(-w), torch.log(w)
    log_x = torch.logaddexp(log_a + F.logsigmoid(l_a), log_b + F.logsigmoid(l_b))
    log_1mx = torch.logaddexp(log_a + F.logsigmoid(-l_a), log_b + F.logsigmoid(-l_b))
    return torch.cat([log_x - log_1mx, out[:, 1:2]], 1)


def node_output(out, w, node):
    """The network output with channel 0 the x_HI logit, for every head."""
    return two_node_output(out, w, node["monotone"]) if node and node["two_node"] else out


def tb_normalized(out, phys, head, tb_offset, tb_scale):
    """Normalized T_b from the network output (B, 2, H, W)."""
    if head == "plain":
        return out[:, 1:2]
    tb = out[:, :1].sigmoid() * phys * (1.0 - torch.exp(out[:, 1:2].clamp(max=U_MAX)))
    return (tb - tb_offset) / tb_scale


@torch.no_grad()
def evaluate(model, loader, device, head, tb_norm, collect=False, power=None, node=None):
    model.eval()
    sums = {"bce": 0.0, "x_se": 0.0, "t_se": 0.0, "n": 0, "p_sum": 0.0, "p_n": 0, "px_sum": 0.0}
    kept = {"x_pred": [], "x_true": [], "t_pred": [], "t_true": [], "z": [], "cone_id": []}
    for batch in loader:
        w = batch.get("w")
        out = node_output(model(batch["cond"], node_scalars(batch["scalars"], w, node)), w, node)
        t = tb_normalized(out, batch["phys"], head, *tb_norm)
        x, y = out[:, :1].sigmoid(), batch["target"]
        sums["bce"] += float(F.binary_cross_entropy_with_logits(out[:, :1], y[:, :1], reduction="sum"))
        sums["x_se"] += float((x - y[:, :1]).square().sum())
        sums["t_se"] += float((t - y[:, 1:]).square().sum())
        sums["n"] += y[:, :1].numel()
        if power is not None:
            sums["p_sum"] += float(power.per_slice(t, y[:, 1:]).sum())
            sums["px_sum"] += float(power.per_slice(x, y[:, :1]).sum())
            sums["p_n"] += len(t)
        if collect:
            for k, v in (("x_pred", x), ("x_true", y[:, :1]), ("t_pred", t), ("t_true", y[:, 1:])):
                kept[k].append(v[:, 0].half().cpu().numpy())
            kept["z"].append(batch["z"].numpy())
            kept["cone_id"].append(batch["cone_id"].numpy())
    n = sums["n"]
    x_mse, t_nmse = sums["x_se"] / n, sums["t_se"] / n
    scores = {"bce": sums["bce"] / n, "xhi_rmse": math.sqrt(x_mse), "tb_rmse_mk": math.sqrt(t_nmse) * tb_norm[1],
              "xhi_nmse": x_mse, "tb_nmse": t_nmse, "score": 0.5 * (x_mse + t_nmse)}
    if power is not None:
        scores["tb_power_logerr"] = sums["p_sum"] / max(sums["p_n"], 1)
        scores["xhi_power_logerr"] = sums["px_sum"] / max(sums["p_n"], 1)
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
    node = node_phase(config)
    w_mode = node and node["w_mode"]
    t0 = time.perf_counter()
    train = MFMemorySplit(cache_path, "train", storage, device, w_mode=w_mode)
    val = MFMemorySplit(cache_path, "val", storage, device, w_mode=w_mode)
    print(f"Loaded train/val: {(train.nbytes + val.nbytes) / 2**30:.1f} GiB, {len(train)}/{len(val)} rows, "
          f"{time.perf_counter() - t0:.0f} s", flush=True)
    norm = json.loads(train.attrs["normalization"])["brightness_temp"]
    tb_norm = (float(norm["offset"]), float(norm["scale"]))
    keep = scalar_keep(json.loads(train.attrs["scalar_names"]), config.drop_history)
    if config.drop_history:
        for split in (train, val):
            split.scalars = split.scalars[:, keep]
        print(f"drop_history={config.drop_history}: {len(keep)} of {len(json.loads(train.attrs['scalar_names']))} "
              f"scalars kept", flush=True)
    eval_batch = config.eval_batch_size or config.batch_size
    stream = MemoryStream(train, config.batch_size, config.seed)
    val_loader = val.loader(eval_batch)
    model_config = {"in_channels": train.cond.shape[1], "out_channels": 3 if node and node["two_node"] else 2,
                    "scalar_dim": train.scalars.shape[1] + bool(node and node["input"]),
                    "width": config.width, "modes": config.modes, "n_steps": config.updates,
                    "step_size": config.step_size}
    if config.radius:
        model_config.update(radius=config.radius, rank=config.rank or None)
    def parse_range(text, name):
        if not text:
            return None
        lo_hi = tuple(int(v) for v in text.split(":"))
        if len(lo_hi) != 2 or not 1 <= lo_hi[0] <= lo_hi[1]:
            raise ValueError(f"{name} must be 'lo:hi' with 1 <= lo <= hi")
        return lo_hi
    t_range = parse_range(config.random_updates, "random_updates")
    tc_range = parse_range(config.random_coarse_updates, "random_coarse_updates")
    if tc_range and not config.coarse_factor:
        raise ValueError("random_coarse_updates needs the two-level model (--coarse-factor)")
    if config.gate_mode != "full":
        if config.coarse_factor:
            raise ValueError("gate ablations are implemented for the single-level model")
        model_config["gate_mode"] = config.gate_mode
    if bool(config.disc_radius) != bool(config.disc_pathways):
        raise ValueError("disc_radius and disc_pathways must be set together")
    if config.disc_norm and not config.disc_pathways:
        raise ValueError("disc_norm needs disc pathways")
    if (config.pathways, config.kernel_groups, config.share_pathways, config.disc_pathways) != (1, 1, "", 0):
        if config.coarse_factor or config.radius:
            raise ValueError("multi-pathway interactions are implemented for the single-level spectral model")
        model_config.update(pathways=config.pathways, kernel_groups=config.kernel_groups,
                            share=config.share_pathways)
        if config.disc_pathways:
            model_config.update(disc_radius=config.disc_radius, disc_pathways=config.disc_pathways)
            if config.disc_norm:
                model_config["disc_norm"] = config.disc_norm
    if config.coarse_factor:
        h, w = train.cond.shape[-2:]
        model_config.update(kind="unet", factor=config.coarse_factor, coarse_steps=config.coarse_updates,
                            coarse_modes=config.coarse_modes or min(h, w) // config.coarse_factor // 2,
                            coarse_radius=config.coarse_radius or None, coarse_rank=config.coarse_rank or None)
    if config.radius or config.coarse_radius:
        torch.backends.cudnn.benchmark = True  # cuDNN's default algorithm choice is erratic for wide stencils
    model = build_model(model_config).to(device)
    power = PowerLoss(*train.cond.shape[-2:], config.power_kmin, config.power_bins, config.power_eps).to(device)
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
        scores = evaluate(model, val_loader, device, config.head, tb_norm, power=power, node=node)
        plateau, best_model = PlateauState(scores["score"], scores["score"]), _cpu_state(model)
        history.append({"step": 0, **{f"val_{k}": v for k, v in scores.items()}, "lr": config.lr})
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metadata.json").write_text(json.dumps({
            **signature, "model_config": model_config, "cache": str(Path(cache_path).resolve()),
            "cache_attrs": {k: (v.item() if hasattr(v, "item") else v) for k, v in train.attrs.items()},
            "tb_normalization": tb_norm, "git_commit": git_commit(root), "scalar_keep": keep,
            "rows": {"train": len(train), "val": len(val)},
            "objective": "BCE(x_HI) + tb_weight * MSE(normalized T_b); selection: mean normalized MSE",
            "initial_max_steps": config.max_steps, **({"node_phase": node} if node else {})}, indent=2))
    start = time.perf_counter()

    def save():
        saved = elapsed + time.perf_counter() - start
        _atomic_save({"signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "model_config": model_config, "stream": stream.state_dict(), "torch_rng": torch.get_rng_state(),
                      "step": step, "samples_seen": samples, "elapsed_seconds": saved, "plateau": asdict(plateau),
                      "best_model": best_model, "history": history, "stopped": stopped}, run_dir / "last.pt")
        _atomic_save({"model": best_model, "model_config": model_config, "head": config.head, "tb_norm": tb_norm,
                      "step": plateau.best_step, "validation_score": plateau.best,
                      **({"scalar_keep": keep} if config.drop_history else {}),
                      **({"node_phase": node} if node else {})}, run_dir / "best.pt")
        (run_dir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
        return saved

    save()
    while step < config.max_steps and not stopped:
        model.train()
        batch = stream.next()
        y = batch["target"]
        optimizer.zero_grad(set_to_none=True)
        rng = np.random.default_rng([config.seed, step])
        steps = {}
        if t_range:
            steps["n_steps"] = int(rng.integers(t_range[0], t_range[1] + 1))
        if tc_range:
            steps["coarse_steps"] = int(rng.integers(tc_range[0], tc_range[1] + 1))
        w = batch.get("w")
        out = node_output(model(batch["cond"], node_scalars(batch["scalars"], w, node), **steps), w, node)
        t = tb_normalized(out, batch["phys"], config.head, *tb_norm)
        loss = F.binary_cross_entropy_with_logits(out[:, :1], y[:, :1]) + config.tb_weight * F.mse_loss(t, y[:, 1:])
        if config.power_weight:
            loss = loss + config.power_weight * power(t, y[:, 1:])
        if config.xhi_power_weight:
            loss = loss + config.xhi_power_weight * power(out[:, :1].sigmoid(), y[:, :1])
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
            scores = evaluate(model, val_loader, device, config.head, tb_norm, power=power, node=node)
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
                  f"val x_HI {scores['xhi_rmse']:.5f} T_b {scores['tb_rmse_mk']:.3f} mK P_err {scores.get('tb_power_logerr', 0):.4f} Px_err {scores.get('xhi_power_logerr', 0):.4f} "
                  f"score {scores['score']:.6f} "
                  f"best {plateau.best:.6f} lr {lr:.2g}->{next_lr:.2g} plateau {plateau.bad_checks}/{config.stop_patience}",
                  flush=True)
    seconds = save()
    result = {"stop_reason": "validation_plateau_at_min_lr" if stopped else "max_steps", "steps": step,
              "samples_seen": samples, "epochs_seen": samples / len(train), "best_step": plateau.best_step,
              "best_validation_score": plateau.best, "elapsed_training_seconds": seconds}
    if stopped:
        model.load_state_dict(best_model)
        test = MFMemorySplit(cache_path, "test", storage, device, w_mode=w_mode)
        test.scalars = test.scalars[:, keep]
        scores, kept = evaluate(model, test.loader(eval_batch), device, config.head, tb_norm, collect=True, power=power,
                                node=node)
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
        kind = {"action": argparse.BooleanOptionalAction} if isinstance(f.default, bool) else {"type": type(f.default)}
        p.add_argument("--" + f.name.replace("_", "-"), default=f.default, **kind)
    a = p.parse_args(argv)
    config = MFConfig(**{f.name: getattr(a, f.name) for f in fields(MFConfig)
                         if f.name not in ("data", "channels", "train_cones")})
    run(a.cache, a.run_dir, config, a.device, a.resume)


if __name__ == "__main__":
    main()
