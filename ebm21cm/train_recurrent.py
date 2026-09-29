"""Train and compare recurrent field operators on an existing slice cache.

This is a deterministic x_HI experiment, not an EBM or an ensemble sampler.
Binary cross entropy also accepts fractional x_HI targets. With incomplete
conditioning its optimum is still a conditional mean; sharp predictions are
not evidence of a calibrated generative distribution.

    python -m ebm21cm.train_recurrent --cache slices.h5 --run-dir runs/recurrent
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
from torch.utils.data import DataLoader, Subset

from .data.cache import SliceDataset, read_stats
from .model.recurrent import RecurrentFNO2d
from .model.rhizome import RhizomeOperator2d
from .run import git_commit


MODEL_TYPES = {"additive": RecurrentFNO2d, "rhizome": RhizomeOperator2d}
VARIANTS = {
    "rhizome": {"architecture": "rhizome"},
    "rhizome_untied": {"architecture": "rhizome", "untied": True},
    "recurrent": {},
    "untied": {"untied": True},
    "local_only": {"use_spectral": False},
    "spectral_only": {"use_local": False},
    "pointwise": {"use_local": False, "use_spectral": False},
}


@dataclass
class ExperimentConfig:
    steps: int = 400
    batch_size: int = 16
    width: int = 16
    modes: int = 4
    updates: int = 4
    step_size: float = 0.5
    lr: float = 0.003
    val_every: int = 100
    seed: int = 0
    threads: int = 2

    def validate(self):
        for key in ("steps", "batch_size", "width", "modes", "updates", "val_every", "threads"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not 0 < self.step_size <= 1:
            raise ValueError("step_size must be in (0, 1]")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


def _batch(batch, device):
    cond, scalars = (batch[k].to(device) for k in ("cond", "scalars"))
    # The shared cache uses 2*x_HI-1 as its first target channel.
    target = (batch["target"][:, :1].to(device) + 1) / 2
    if not torch.isfinite(cond).all() or not torch.isfinite(scalars).all():
        raise ValueError("cache conditioning must be finite")
    if not torch.isfinite(target).all() or torch.any((target < 0) | (target > 1)):
        raise ValueError("cache x_HI targets must be finite and in [0, 1]")
    return cond, scalars, target


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def validation_bce(model, loader, device):
    model.eval()
    total, pixels = 0.0, 0
    for batch in loader:
        cond, scalars, truth = _batch(batch, device)
        total += float(F.binary_cross_entropy_with_logits(model(cond, scalars), truth, reduction="sum"))
        pixels += truth.numel()
    return total / pixels


def field_metrics(pred, truth):
    """Physical-unit metrics; report mixed slices separately to expose trivial phases."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    if pred.shape != truth.shape or pred.ndim != 3 or len(pred) == 0:
        raise ValueError("pred and truth must have matching nonempty (N,H,W) shapes")

    def summarize(p, t):
        if len(t) == 0:
            return {"n_slices": 0}
        pi, ti = p < 0.5, t < 0.5
        union = np.logical_or(pi, ti).sum()
        return {
            "n_slices": len(t),
            "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            "pixel_accuracy": float(np.mean(pi == ti)),
            "ionized_iou": float(np.logical_and(pi, ti).sum() / union) if union else None,
            "slice_mean_mae": float(np.mean(np.abs(p.mean((1, 2)) - t.mean((1, 2))))),
            "hedged_fraction": float(np.mean((p > 0.1) & (p < 0.9))),
            "truth_hedged_fraction": float(np.mean((t > 0.1) & (t < 0.9))),
        }

    fraction = truth.mean((1, 2))
    mixed = (fraction > 0.05) & (fraction < 0.95)
    return {"all": summarize(pred, truth), "mixed": summarize(pred[mixed], truth[mixed])}


@torch.no_grad()
def predict(model, loader, device, n_steps=None):
    model.eval()
    fields = {key: [] for key in ("prediction", "truth", "row", "cone_id", "z")}
    start = time.perf_counter()
    for batch in loader:
        cond, scalars, truth = _batch(batch, device)
        fields["prediction"].append(model(cond, scalars, n_steps=n_steps).sigmoid()[:, 0].cpu().numpy())
        fields["truth"].append(truth[:, 0].cpu().numpy())
        for key in ("row", "cone_id", "z"):
            fields[key].append(batch[key].numpy())
    _sync(device)
    seconds = time.perf_counter() - start
    fields = {key: np.concatenate(value) for key, value in fields.items()}
    return fields, seconds


def load_checkpoint(path, device="cpu"):
    """Load a trained operator; checkpoint also carries the cache normalization."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    # Checkpoints from the original experiment had no architecture tag.
    architecture = checkpoint.get("architecture", "additive")
    if architecture not in MODEL_TYPES:
        raise ValueError(f"unknown checkpoint architecture {architecture!r}")
    model = MODEL_TYPES[architecture](**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def _example_rows(truth):
    # Prefer mixed-phase slices, but retain a useful figure for small smoke tests.
    means = truth.mean((1, 2))
    candidates = np.flatnonzero((means > 0.05) & (means < 0.95))
    if len(candidates) == 0:
        candidates = np.arange(len(truth))
    return candidates[np.unique(np.linspace(0, len(candidates) - 1, min(4, len(candidates))).astype(int))]


def _plot_examples(path, predictions, center_density):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = next(iter(predictions.values()))
    truth = first["truth"]
    rows = range(len(truth))
    names = ["density", "truth", *predictions]
    fig, axes = plt.subplots(len(rows), len(names), figsize=(3 * len(names), 3 * len(rows)), squeeze=False)
    for i, row in enumerate(rows):
        images = [center_density[row], truth[row], *(p["prediction"][row] for p in predictions.values())]
        for j, image in enumerate(images):
            kw = {"cmap": "RdBu_r"} if j == 0 else {"cmap": "viridis", "vmin": 0, "vmax": 1}
            axes[i, j].imshow(image, origin="lower", **kw)
            axes[i, j].set_title(f"{names[j]}, z={first['z'][row]:.2f}")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def _rollout_probe(model, dataset, rows, device, max_steps, path):
    """Track hidden-state changes and decoded fields beyond the training horizon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batch = next(iter(DataLoader(Subset(dataset, rows), batch_size=len(rows))))
    cond, scalars, truth = _batch(batch, device)
    # Decoding intermediate states does not feed the decoded field back into the model.
    _, states = model(cond, scalars, n_steps=max_steps, return_states=True)
    residual = [float((b - a).square().mean().sqrt()) for a, b in zip(states[:-1], states[1:])]
    # Use the public forward interface rather than depending on decoder internals.
    show_steps = sorted({1, model.n_steps, max_steps})
    images = [truth[0, 0].cpu().numpy()]
    images += [model(cond[:1], scalars[:1], n_steps=s).sigmoid()[0, 0].cpu().numpy() for s in show_steps]
    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3), squeeze=False)
    for ax, image, label in zip(axes[0], images, ["truth", *(f"{s} updates" for s in show_steps)]):
        ax.imshow(image, origin="lower", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"state_update_rms": residual, "max_abs_state": max(float(h.abs().max()) for h in states),
            "cache_rows": batch["row"].tolist(),
            "note": "Bounded states do not establish convergence or accuracy at unseen iteration counts."}


def run_experiment(cache_path, run_dir, config=None, variants=("rhizome", "recurrent"), device="cpu"):
    config = config or ExperimentConfig()
    config.validate()
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError(f"choose distinct variants from {tuple(VARIANTS)}")
    device = torch.device(device)
    # PyTorch's complex FFT path is tested on CPU/CUDA, not MPS.
    if device.type not in ("cpu", "cuda"):
        raise ValueError("use cpu or cuda; this Fourier implementation is not validated on MPS")
    torch.set_num_threads(config.threads)
    stats = read_stats(cache_path)
    datasets = {split: SliceDataset(cache_path, split, stats) for split in ("train", "validation", "test")}
    if any(len(ds) == 0 for ds in datasets.values()):
        raise ValueError("train, validation and test splits must all be nonempty")
    cone_sets = {split: set(ds.cone_id.tolist()) for split, ds in datasets.items()}
    if any(cone_sets[a] & cone_sets[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("cache leaks cone IDs between splits")
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"{run_dir} is not empty; choose a new run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    sources = ("ebm21cm/model/recurrent.py", "ebm21cm/model/rhizome.py", "ebm21cm/train_recurrent.py",
               "ebm21cm/data/cache.py", "ebm21cm/data/toy.py")
    metadata = {
        "config": asdict(config), "variants": list(variants), "cache": str(Path(cache_path).resolve()),
        "stats": stats, "device": str(device), "torch": str(torch.__version__),
        "git_commit": git_commit(root),
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
        "splits": {s: {"rows": len(ds), "cones": sorted(cone_sets[s])} for s, ds in datasets.items()},
        "task": "Deterministic x_HI regression; no stochastic-generation or 21cmFAST accuracy claim.",
        "comparison": "Untied uses the same update count/width, not the same parameter count.",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    val_loader = DataLoader(datasets["validation"], batch_size=config.batch_size)
    test_loader = DataLoader(datasets["test"], batch_size=config.batch_size)
    train_loader = DataLoader(datasets["train"], batch_size=config.batch_size)
    neutral_sum, pixel_count = 0.0, 0
    for batch in train_loader:
        _, _, target = _batch(batch, device)
        neutral_sum += float(target.sum())
        pixel_count += target.numel()
    neutral_mean = neutral_sum / pixel_count
    result, predictions = {}, {}
    example_rows = None
    for variant in variants:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        train_ds = SliceDataset(cache_path, "train", stats, augment_data=True, seed=config.seed)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False,
                                  generator=torch.Generator().manual_seed(config.seed))
        options = VARIANTS[variant].copy()
        architecture = options.pop("architecture", "additive")
        model_config = {
            "in_channels": train_ds.n_cond, "out_channels": 1, "scalar_dim": train_ds.n_scalars,
            "width": config.width, "modes": config.modes, "n_steps": config.updates,
            "step_size": config.step_size, **options,
        }
        model = MODEL_TYPES[architecture](**model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        variant_dir = run_dir / variant
        variant_dir.mkdir()
        initial = validation_bce(model, val_loader, device)
        best, best_step = initial, 0

        def save_best(step, loss):
            path = variant_dir / "best.pt"
            torch.save({"model": model.state_dict(), "model_config": model_config,
                        "architecture": architecture,
                        "stats": stats, "step": step, "validation_bce": loss}, path.with_suffix(".pt.tmp"))
            path.with_suffix(".pt.tmp").replace(path)

        save_best(0, initial)
        iterator = iter(train_loader)
        start = time.perf_counter()
        history = []
        with (variant_dir / "metrics.jsonl").open("w") as log:
            for step in range(1, config.steps + 1):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                model.train()
                cond, scalars, target = _batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(cond, scalars), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{variant}: non-finite loss at step {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if step % config.val_every == 0 or step == config.steps:
                    val = validation_bce(model, val_loader, device)
                    if val < best:
                        best, best_step = val, step
                        save_best(step, val)
                    record = {"step": step, "train_bce": float(loss.detach()), "validation_bce": val,
                              "grad_norm": float(norm)}
                    history.append(record)
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(f"{variant:13s} step {step:5d}: train={record['train_bce']:.4f} val={val:.4f}", flush=True)
        _sync(device)
        train_seconds = time.perf_counter() - start
        model, _ = load_checkpoint(variant_dir / "best.pt", device)
        fields, prediction_seconds = predict(model, test_loader, device)
        if example_rows is None:
            example_rows = _example_rows(fields["truth"])
        # Keep only plotting rows across variants, not multiple full test sets.
        predictions[variant] = {k: fields[k][example_rows] for k in ("prediction", "truth", "z")}
        np.savez_compressed(variant_dir / "test_predictions.npz", **fields)
        report = {
            "architecture": architecture,
            "parameters_real": sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()),
            "initial_validation_bce": initial, "best_validation_bce": best, "best_step": best_step,
            "train_seconds_including_validation": train_seconds,
            "test_seconds_including_io": prediction_seconds,
            "test": field_metrics(fields["prediction"], fields["truth"]), "history": history,
        }
        if not model.untied:
            counts = sorted({1, max(1, config.updates // 2), config.updates, 2 * config.updates})
            report["iteration_sweep"] = {}
            for count in counts:
                sweep, _ = predict(model, test_loader, device, n_steps=count)
                report["iteration_sweep"][str(count)] = field_metrics(sweep["prediction"], sweep["truth"])
                del sweep
            means = fields["truth"].mean((1, 2))
            probe_rows = np.flatnonzero((means > 0.05) & (means < 0.95))
            if len(probe_rows) == 0:
                probe_rows = np.arange(len(means))
            report["rollout"] = _rollout_probe(model, datasets["test"], probe_rows[:4].tolist(),
                                               device, 2 * config.updates, variant_dir / "rollout.png")
        result[variant] = report
        if "constant_train_mean" not in result:
            truth = fields["truth"]
            result["constant_train_mean"] = {
                "value": neutral_mean, "test": field_metrics(np.full_like(truth, neutral_mean), truth),
            }
            del truth
        del fields
    (run_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    center = datasets["test"].center_band
    density = np.stack([datasets["test"].raw(int(i))["delta"][center] for i in example_rows])
    _plot_examples(run_dir / "examples.png", predictions, density)
    print(f"Wrote {run_dir / 'results.json'}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=["rhizome", "recurrent"])
    parser.add_argument("--device", default="cpu")
    defaults = ExperimentConfig()
    for key, value in asdict(defaults).items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args(argv)
    config = ExperimentConfig(**{key: getattr(args, key) for key in asdict(defaults)})
    run_experiment(args.cache, args.run_dir, config, args.variants, args.device)


if __name__ == "__main__":
    main()
