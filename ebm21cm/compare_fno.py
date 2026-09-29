"""Side-by-side rhizome vs fno-21cm 2-D x_HI predictions on identical slices.

    python -m ebm21cm.compare_fno --rhizome runs/sweep1/w128 \\
        --fno LWF=$FNO/checkpoints/2d_xhi/fno_lwf/xhi2d_lwf_glob_e100 ... \\
        --out runs/compare_fno/w128

Every model predicts the same cache rows. fno-21cm's inputs are rebuilt from
the row exactly as its ``SliceCache`` does (density/10, 1/(1+z), normalized
parameters broadcast): the cache's single-slice band at offset 0 *is* the
native density slice fno-21cm reads (float16 here). Rows come from cones that
no model trained on: by default the rhizome validation split, restricted to
cones in every fno run's test split, and to fno-21cm's training window
0.05 < mean x_HI < 0.95.

Figures follow fno-21cm's 2-D viz (viridis x_HI, RdBu_r signed errors,
representative redshift quantiles). Imports fno-21cm from ``--fno-root``;
the rest of ebm21cm does not depend on it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import metrics  # noqa: E402
from .data.cache import SPLITS  # noqa: E402
from .data.memory import MemorySplit  # noqa: E402
from .train_recurrent import load_checkpoint as load_rhizome  # noqa: E402

RHIZOME = "Rhizome"
COLORS = ["#264653", "#e07a5f", "#168aad", "#f4a261", "#7b2cbf", "#2a9d8f", "#e9c46a"]
QUANTILES = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]


def fno_modules(root):
    """Import fno-21cm's run loader, model factory and parameter normalization."""
    sys.path.insert(0, str(Path(root).resolve()))
    from util.neuralop_setup import prefer_local_neuralop

    prefer_local_neuralop()
    from dataset.dataset_3d import ParameterNormalization
    from modeling import load_checkpoint
    from viz.compare_xhi2d_models import build_model, load_run

    return load_run, build_model, load_checkpoint, ParameterNormalization


def select_rows(cache, split, cones, window):
    with h5py.File(cache, "r") as h:
        rows = np.flatnonzero(h["split"][:] == SPLITS[split])
        rows = rows[np.isin(h["cone_id"][:][rows], list(cones))]
        means = np.array([h["xhi"][r].astype(np.float64).mean() for r in rows])
    keep = (means > window[0]) & (means < window[1])
    return rows[keep], means[keep]


@torch.inference_mode()
def predict_rhizome(run_dir, data, positions, device, batch):
    model, checkpoint = load_rhizome(Path(run_dir) / "best.pt", device)
    out = []
    for a in range(0, len(positions), batch):
        b = data.batch(positions[a:a + batch])
        out.append(model(b["cond"], b["scalars"]).sigmoid()[:, 0].cpu().numpy())
    return np.concatenate(out), checkpoint


@torch.inference_mode()
def predict_fno(run, build_model, load_checkpoint, norm_cls, density, z, params, device, batch):
    model = build_model(run.metadata["model_config"], run.metadata)
    result = load_checkpoint(model, run.checkpoint)
    # Do not require matched == total: fno-21cm migrates some older checkpoints
    # inside _load_from_state_dict (e.g. learned-waveform phase_weight is
    # zero-filled, which is exact), so pre-load key/shape counts undercount.
    if result.missing or result.unexpected:
        raise RuntimeError(f"incomplete checkpoint load for {run.label}: "
                           f"missing={result.missing}, unexpected={result.unexpected}")
    model = model.to(device).eval()
    features = run.metadata["input_features"]["name"]
    if features != "density_z_params":
        raise ValueError(f"{run.label}: only density_z_params inputs are rebuilt, got {features}")
    p = norm_cls.from_dict(run.metadata["parameter_normalization"]).normalize(params.astype(np.float32))
    out = []
    shape = density.shape[-2:]
    for a in range(0, len(density), batch):
        d = torch.from_numpy(density[a:a + batch] / 10.0).float()
        scal = np.concatenate([1.0 / (1.0 + z[a:a + batch, None]), p[a:a + batch]], 1).astype(np.float32)
        x = torch.cat([d[:, None], torch.from_numpy(scal)[:, :, None, None].expand(-1, -1, *shape)], 1)
        out.append(model(x.to(device)).cpu().numpy()[:, 0])
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return np.clip(np.concatenate(out), 0.0, 1.0)


def representative(z, means, quantiles):
    """fno-21cm's rule: nearest slices to each z quantile, then the one at the local median x_HI."""
    chosen, used = [], set()
    local = max(32, len(z) // 100)
    for target in np.quantile(z, quantiles):
        near = np.argsort(np.abs(z - target), kind="stable")[:local]
        ranked = near[np.argsort(np.abs(means[near] - np.median(means[near])), kind="stable")]
        pick = next(int(i) for i in ranked if int(i) not in used)
        chosen.append(pick)
        used.add(pick)
    return chosen


def slice_scores(pred, truth, cell):
    err = pred - truth
    pi, ti = pred < 0.5, truth < 0.5
    union = (pi | ti).sum((1, 2))
    return {
        "rmse": np.sqrt((err ** 2).mean((1, 2))),
        "mean_err": pred.mean((1, 2)) - truth.mean((1, 2)),
        "iou": np.where(union > 0, (pi & ti).sum((1, 2)) / np.maximum(union, 1), np.nan),
        "hedged": ((pred > 0.1) & (pred < 0.9)).mean((1, 2)),
    }


def summary(pred, truth, cell, edges, bsd_rows):
    s = slice_scores(pred, truth, cell)
    with np.errstate(invalid="ignore", divide="ignore"):
        pab, paa, pbb = metrics.cross_power(pred, truth, cell, edges)
        ratio, r = np.nanmean(paa / pbb, 0), np.nanmean(pab / np.sqrt(paa * pbb), 0)
    sizes = np.concatenate([metrics.mfp_sizes(pred[i] < 0.5, cell, 500, np.random.default_rng(i)) for i in bsd_rows])
    return s, ratio, r, sizes


def field_grid(path, fields, names, info, title, cmap, vmin, vmax, label, annotate=None, truth=None):
    rows, cols = len(info), len(names)
    fig, axes = plt.subplots(rows, cols, figsize=(2.55 * cols + 1.2, 2.6 * rows),
                             constrained_layout=True, squeeze=False)
    image = None
    for i, meta in enumerate(info):
        for j, name in enumerate(names):
            ax = axes[i, j]
            image = ax.imshow(fields[name][i], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                              interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=10)
            if annotate is not None and name in annotate:
                rmse = float(np.sqrt(np.mean((fields[name][i] - truth[i]) ** 2))) if truth is not None \
                    else float(np.sqrt(np.mean(fields[name][i] ** 2)))
                ax.text(0.02, 0.98, f"RMSE={rmse:.3f}", transform=ax.transAxes, va="top", ha="left",
                        fontsize=8, color="white", bbox={"facecolor": "black", "alpha": 0.55, "pad": 2})
        axes[i, 0].set_ylabel(f"z={meta['z']:.2f}  cone={meta['cone_id']}\nmean $x_{{HI}}$={meta['xhi_mean']:.2f}",
                              fontsize=9)
    fig.colorbar(image, ax=axes, shrink=0.75, label=label)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def edge_zoom(path, truth, preds, names, info, size=48):
    """Crops of maximal front density, truth x_HI=0.5 contour overlaid on every panel."""
    rows = [i for i, m in enumerate(info) if 0.2 < m["xhi_mean"] < 0.8] or list(range(len(info)))
    fig, axes = plt.subplots(len(rows), 1 + len(names), figsize=(2.4 * (1 + len(names)) + 1.0, 2.5 * len(rows)),
                             constrained_layout=True, squeeze=False)
    image = None
    for a, i in enumerate(rows):
        t = truth[i]
        edges = np.abs(np.diff(t, axis=0, append=t[:1])) + np.abs(np.diff(t, axis=1, append=t[:, :1]))
        density = np.cumsum(np.cumsum(np.pad(edges, ((1, 0), (1, 0))), 0), 1)
        n = t.shape[0] - size
        scores = (density[size:, size:] - density[:-size, size:] - density[size:, :-size] + density[:-size, :-size])
        y0, x0 = np.unravel_index(np.argmax(scores[:n + 1, :n + 1]), (n + 1, n + 1))
        crop = (slice(y0, y0 + size), slice(x0, x0 + size))
        for j, (name, field) in enumerate([("Truth", t)] + [(k, preds[k][i]) for k in names]):
            ax = axes[a, j]
            image = ax.imshow(field[crop], origin="lower", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            ax.contour(t[crop], levels=[0.5], colors="white", linewidths=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if a == 0:
                ax.set_title(name, fontsize=10)
        axes[a, 0].set_ylabel(f"z={info[i]['z']:.2f}\nmean={info[i]['xhi_mean']:.2f}\n"
                              f"crop ({y0},{x0}) {size}px", fontsize=9)
    fig.colorbar(image, ax=axes, shrink=0.75, label="$x_{HI}$  (white: truth $x_{HI}$=0.5)")
    fig.suptitle("Ionization fronts: densest-edge crops of representative slices", fontsize=13)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rhizome", required=True, help="train_rhizome run directory with best.pt")
    parser.add_argument("--rhizome-label", default=None)
    parser.add_argument("--fno", nargs="+", required=True, help="LABEL=fno-21cm 2-D checkpoint dir")
    parser.add_argument("--fno-root", default="/pfs/10/work/hd_id260-fno_training/fno-21cm")
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--window", type=float, nargs=2, default=(0.05, 0.95))
    parser.add_argument("--bsd-slices", type=int, default=300)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        parser.error("comparing on the rhizome test split requires --allow-test")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    load_run, build_model, load_fno_checkpoint, norm_cls = fno_modules(args.fno_root)

    runs = []
    for spec in args.fno:
        label, path = spec.split("=", 1)
        run = load_run(label, Path(path))
        if run is None:
            raise SystemExit(f"cannot load fno run {spec}")
        runs.append(run)
    rhizome_dir = Path(args.rhizome)
    meta = json.loads((rhizome_dir / "metadata.json").read_text())
    cache = meta["cache"]
    cones = set(meta["splits"][args.split]["cones"])
    for run in runs:
        cones &= set(run.metadata["split"]["test_cone_ids"])
    rows, means = select_rows(cache, args.split, cones, args.window)
    print(f"{len(rows)} slices from {len(cones)} cones held out by every model", flush=True)

    _, checkpoint = load_rhizome(rhizome_dir / "best.pt", "cpu")
    data = MemorySplit(cache, args.split, checkpoint["stats"], checkpoint.get("channels", "all"),
                       cones=sorted(cones), storage="cpu", device=args.device)
    positions = np.searchsorted(data.rows, rows)
    assert np.array_equal(data.rows[positions], rows)
    with h5py.File(cache, "r") as h:
        cell = float(h.attrs["cell_size_mpc"])
        bands = json.loads(h.attrs["bands"])
        center = [j for j, (lo, hi) in enumerate(bands) if lo == 0 and hi == 1][0]
        density = np.stack([h["delta"][r, center] for r in rows]).astype(np.float32)
        truth = np.stack([h["xhi"][r] for r in rows]).astype(np.float32)
        z = h["z"][:][rows]
        params = h["params"][:][rows]
        cone_id = h["cone_id"][:][rows]

    label = args.rhizome_label or f"{RHIZOME} ({rhizome_dir.name})"
    preds = {}
    preds[label], _ = predict_rhizome(rhizome_dir, data, positions, args.device, args.batch)
    for run in runs:
        preds[run.label] = predict_fno(run, build_model, load_fno_checkpoint, norm_cls, density, z, params,
                                       args.device, args.batch)
        print(f"predicted {run.label}", flush=True)
    names = list(preds)
    colors = dict(zip(names, COLORS))

    # ------------------------------------------------------------ aggregate metrics
    edges = metrics.k_bins(truth.shape[-2:], cell)
    kc = 0.5 * (edges[1:] + edges[:-1])
    bsd_rows = np.random.default_rng(0).choice(len(rows), min(args.bsd_slices, len(rows)), replace=False)
    truth_sizes = np.concatenate([metrics.mfp_sizes(truth[i] < 0.5, cell, 500, np.random.default_rng(i))
                                  for i in bsd_rows])
    report, per_slice, curves = {}, {}, {}
    for name in names:
        s, ratio, r, sizes = summary(preds[name], truth, cell, edges, bsd_rows)
        per_slice[name], curves[name] = s, (ratio, r, sizes)
        high = kc > 0.5 * kc.max()
        report[name] = {
            "slice_rmse_mean": float(s["rmse"].mean()),
            "pooled_rmse": float(np.sqrt(np.mean((preds[name] - truth) ** 2))),
            "mean_xhi_mae": float(np.abs(s["mean_err"]).mean()),
            "ionized_iou": float(np.nanmean(s["iou"])),
            "hedged_fraction": float(s["hedged"].mean()),
            "power_ratio_high_k": float(np.nanmean(ratio[high])),
            "cross_correlation_high_k": float(np.nanmean(r[high])),
            "bsd_w1_log_mfp": metrics.size_distance(sizes, truth_sizes),
        }
    report_all = {
        "rows": len(rows), "cones": len(cones), "split": args.split, "window": list(args.window),
        "truth_hedged_fraction": float(((truth > 0.1) & (truth < 0.9)).mean()),
        "rhizome_checkpoint": {"run": str(rhizome_dir.resolve()), "step": checkpoint["step"],
                               "validation_bce": checkpoint["validation_bce"]},
        "fno_runs": {run.label: {"path": str(run.path.resolve()), "reported_test_rmse": run.report["test_rmse"],
                                 "reported_val_rmse": run.report["val_rmse"]} for run in runs},
        "high_k_threshold_per_mpc": float(0.5 * kc.max()),
        "models": report,
    }
    (out / "metrics_comparison.json").write_text(json.dumps(report_all, indent=2) + "\n")
    with (out / "per_slice.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "cone_id", "z", "xhi_mean", *[f"rmse[{n}]" for n in names]])
        for i in range(len(rows)):
            writer.writerow([int(rows[i]), int(cone_id[i]), f"{z[i]:.4f}", f"{means[i]:.4f}",
                             *[f"{per_slice[n]['rmse'][i]:.5f}" for n in names]])

    # ------------------------------------------------------------ representative slices
    picks = representative(z, means, QUANTILES)
    info = [{"z": float(z[i]), "cone_id": int(cone_id[i]), "xhi_mean": float(means[i]), "row": int(rows[i])}
            for i in picks]
    t_rep = truth[picks]
    fields = {"Truth": t_rep, **{n: preds[n][picks] for n in names}}
    field_grid(out / "representative_z_predictions.png", fields, ["Truth", *names], info,
               "Held-out representative redshift slices (identical inputs)", "viridis", 0, 1, "$x_{HI}$",
               annotate=names, truth=t_rep)
    errors = {n: preds[n][picks] - t_rep for n in names}
    limit = max(0.05, float(np.quantile(np.abs(np.concatenate([e.ravel() for e in errors.values()])), 0.99)))
    field_grid(out / "representative_z_errors.png", errors, names, info,
               "Signed errors on representative redshift slices", "RdBu_r", -limit, limit,
               "prediction - truth", annotate=names)
    edge_zoom(out / "edge_zoom.png", t_rep, {n: preds[n][picks] for n in names}, names, info)
    (out / "selected_slices.json").write_text(json.dumps(info, indent=2) + "\n")

    # ------------------------------------------------------------ metric bars
    panels = (("slice_rmse_mean", "Per-slice RMSE (fno val_l2 def.)", None),
              ("pooled_rmse", "Pooled RMSE", None),
              ("mean_xhi_mae", "Slice-mean $x_{HI}$ MAE", None),
              ("ionized_iou", "Ionized-mask IoU ($x_{HI}$<0.5)", None),
              ("hedged_fraction", "Hedged pixels (0.1<$x_{HI}$<0.9)", report_all["truth_hedged_fraction"]),
              ("power_ratio_high_k", "High-k power ratio", 1.0),
              ("cross_correlation_high_k", "High-k cross-correlation", 1.0),
              ("bsd_w1_log_mfp", "Bubble-size W1 (log MFP)", None))
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    x = np.arange(len(names))
    for ax, (key, title, ref) in zip(axes.flat, panels):
        values = [report[n][key] for n in names]
        bars = ax.bar(x, values, color=[colors[n] for n in names])
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_title(title)
        ax.set_xticks(x, names, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        if ref is not None:
            ax.axhline(ref, color="black", linestyle="--", linewidth=1)
    fig.suptitle(f"{len(rows)} held-out slices, {len(cones)} cones, "
                 f"{args.window[0]} < mean $x_{{HI}}$ < {args.window[1]}", fontsize=13)
    fig.savefig(out / "metrics_comparison.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------ spectra, sizes, error vs phase
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for n in names:
        ratio, r, sizes = curves[n]
        axes[0].plot(kc, ratio, color=colors[n], label=n)
        axes[1].plot(kc, r, color=colors[n], label=n)
        axes[2].hist(np.log10(sizes), bins=40, density=True, histtype="step", color=colors[n], label=n, lw=1.5)
    axes[2].hist(np.log10(truth_sizes), bins=40, density=True, histtype="stepfilled", color="0.8", label="Truth")
    for ax, title, ylabel in ((axes[0], "Power ratio $P_{pred}/P_{true}$", "ratio"),
                              (axes[1], "Cross-correlation $r(k)$", "r(k)")):
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("k [1/Mpc]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[2].set_xlabel("log10 MFP ionized size [Mpc]")
    axes[2].set_title(f"Ionized-region sizes ({len(bsd_rows)} slices)")
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.savefig(out / "spectra_bubble_sizes.png", dpi=160)
    plt.close(fig)

    bins = np.linspace(args.window[0], args.window[1], 10)
    centers = 0.5 * (bins[1:] + bins[:-1])
    which = np.digitize(means, bins) - 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for n in names:
        for ax, key in ((axes[0], "rmse"), (axes[1], "mean_err")):
            v = per_slice[n][key]
            stats = np.array([np.percentile(v[which == b], [25, 50, 75]) if (which == b).any() else [np.nan] * 3
                              for b in range(len(centers))])
            ax.plot(centers, stats[:, 1], "-o", color=colors[n], label=n, ms=4)
            ax.fill_between(centers, stats[:, 0], stats[:, 2], color=colors[n], alpha=0.15)
    axes[0].set_ylabel("per-slice RMSE (median, IQR)")
    axes[1].set_ylabel("predicted - true mean $x_{HI}$")
    axes[1].axhline(0, color="black", linewidth=1)
    for ax in axes:
        ax.set_xlabel("true slice-mean $x_{HI}$")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Error across the reionization phase", fontsize=13)
    fig.savefig(out / "error_vs_phase.png", dpi=160)
    plt.close(fig)

    width = max(len(n) for n in names)
    print(f"{'model':{width}s}  slice_rmse  pooled  mean_mae    iou  hedged  Pk_hi  r_hi   bsd_w1")
    for n in names:
        m = report[n]
        print(f"{n:{width}s}  {m['slice_rmse_mean']:10.4f} {m['pooled_rmse']:7.4f} {m['mean_xhi_mae']:9.4f} "
              f"{m['ionized_iou']:6.3f} {m['hedged_fraction']:7.3f} {m['power_ratio_high_k']:6.3f} "
              f"{m['cross_correlation_high_k']:5.3f} {m['bsd_w1_log_mfp']:8.4f}")
    print(f"truth hedged fraction {report_all['truth_hedged_fraction']:.3f}; wrote {out}")


if __name__ == "__main__":
    main()
