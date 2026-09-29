"""Rhizome 2-D slices vs exported fno-21cm 3-D prediction cones, on the same native planes.

    python -m ebm21cm.compare_fno_cubes --rhizome runs/sweep1/bs32 \\
        --manifest $WORK/data/eval_cubes/<model>/manifest.json \\
        --fno2d LWF=$FNO/checkpoints/2d_xhi/fno_lwf/xhi2d_lwf_glob_e100 --out runs/compare_fno3d/bs32

fno-21cm's ``eval_cubes`` hold predicted/true x_HI on native LOS planes
(``z_native`` are exact lightcone redshifts). For each cone that the rhizome
run never trained on (and that is not in the rhizome test split), up to
``--per-cone`` mixed-phase planes are drawn stratified in mean x_HI; their
13 LOS density bands are read from the raw lightcone, so the rhizome predicts
exactly the plane the 3-D model predicted. Optional 2-D fno runs predict the
same planes from the single density slice. Scores: fno-21cm's boundary-band
edge diagnostic, per-slice RMSE, spectra, and a representative-slice figure.

The comparison is deliberately uneven: the 3-D model sees whole LOS windows
(and whatever extra inputs it was trained with); the rhizome sees +-32 cells.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch

from . import metrics
from .compare_fno import COLORS, field_grid, fno_modules, predict_fno, representative
from .data.cache import Normalizer, band_extent, band_means, parse_bands
from .data.lightcone import Lightcone
from .train_recurrent import load_checkpoint as load_rhizome

RAW = "/pfs/10/work/hd_id260-fno_training/data/data/21cmfast_11d_sample{:06d}.h5"


def pick_planes(entry, per_cone, window, rng_seed):
    """Mixed-phase planes of one exported cone, stratified in mean x_HI."""
    cube = np.load(entry["npz"])
    means = cube["truth"].mean((0, 1))
    ok = np.flatnonzero((means > window[0]) & (means < window[1]))
    if len(ok) > per_cone:
        edges = np.linspace(means[ok].min(), means[ok].max(), per_cone + 1)
        levels = np.random.default_rng(rng_seed).uniform(edges[:-1], edges[1:])
        ok = np.unique(ok[np.abs(means[ok][None] - levels[:, None]).argmin(1)])
    return ok


def load_cone(job):
    """Truth/3-D prediction planes, raw density bands and params for one cone."""
    entry, planes, bands = job
    cube = np.load(entry["npz"])
    below, above = band_extent(bands)
    out = {"cone_id": entry["cone_id"], "planes": [], "truth": [], "pred3d": [], "bands": [], "z": []}
    with Lightcone(RAW.format(entry["cone_id"]), 2**29) as lc:
        params = lc.params()
        for k in planes:
            j = int(np.abs(lc.redshifts - cube["z_native"][k]).argmin())
            if not below <= j < lc.n_los - above:
                continue
            block = lc.read_range("density", j - below, j + above + 1)
            truth = cube["truth"][:, :, k]
            # The exported planes must be the raw native slice itself.
            if not np.array_equal(truth, lc.read_range("neutral_fraction", j, j + 1)[0]):
                raise ValueError(f"cone {entry['cone_id']} plane {k} is not native slice {j}")
            out["planes"].append(int(k))
            out["truth"].append(truth)
            out["pred3d"].append(cube["pred"][:, :, k])
            out["bands"].append(band_means(block, below, bands).astype(np.float32))
            out["z"].append(float(lc.redshifts[j]))
    out["params"] = params
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rhizome", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--label3d", default=None)
    parser.add_argument("--fno2d", nargs="*", default=[], help="LABEL=fno-21cm 2-D checkpoint dir")
    parser.add_argument("--fno-root", default="/pfs/10/work/hd_id260-fno_training/fno-21cm")
    parser.add_argument("--per-cone", type=int, default=16)
    parser.add_argument("--window", type=float, nargs=2, default=(0.05, 0.95))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    load_run, build_model, load_fno_checkpoint, norm_cls = fno_modules(args.fno_root)
    from viz.boundary_band_diagnostic import (
        BandConfig, BoundaryBandAccumulator, front_width, paired_bootstrap, plot_overlay,
    )

    manifest = json.loads(Path(args.manifest).read_text())
    (model3d, entries), = manifest["models"].items()
    label3d = args.label3d or f"fno 3-D ({model3d})"
    rhizome_dir = Path(args.rhizome)
    meta = json.loads((rhizome_dir / "metadata.json").read_text())
    seen = set(meta["train_subset"]["cones"]) if meta.get("train_subset") else set(meta["splits"]["train"]["cones"])
    excluded = seen | set(meta["splits"]["test"]["cones"])
    entries = [e for e in entries if e["cone_id"] not in excluded]
    with h5py.File(meta["cache"], "r") as h:  # all stored bands; the run may use a subset
        bands = [tuple(b) for b in json.loads(h.attrs["bands"])]
    print(f"{len(entries)} exported cones unseen by the rhizome run and outside its test split", flush=True)

    stash = out / "planes.npz"
    if stash.exists():
        data = dict(np.load(stash, allow_pickle=False))
    else:
        jobs = [(e, pick_planes(e, args.per_cone, args.window, e["cone_id"]), bands) for e in entries]
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("fork")) as ex:
            cones = list(ex.map(load_cone, jobs))
        cones = [c for c in cones if c["planes"]]
        data = {
            "cone_id": np.concatenate([[c["cone_id"]] * len(c["planes"]) for c in cones]).astype(np.int64),
            "plane": np.concatenate([c["planes"] for c in cones]).astype(np.int64),
            "z": np.concatenate([c["z"] for c in cones]),
            "truth": np.concatenate([np.stack(c["truth"]) for c in cones]).astype(np.float32),
            "pred3d": np.concatenate([np.stack(c["pred3d"]) for c in cones]).astype(np.float32),
            "bands": np.concatenate([np.stack(c["bands"]) for c in cones]).astype(np.float32),
            "params": np.concatenate([np.tile(c["params"], (len(c["planes"]), 1)) for c in cones]),
        }
        np.savez(stash, **data)
    n = len(data["z"])
    print(f"{n} mixed-phase planes from {len(np.unique(data['cone_id']))} cones", flush=True)

    model, checkpoint = load_rhizome(rhizome_dir / "best.pt", args.device)
    norm = Normalizer(checkpoint["stats"])
    channels = checkpoint.get("channels", list(range(len(bands))))
    cond = norm.cond(data["bands"][:, channels].astype(np.float64)).astype(np.float32)
    scal = norm.scalars(data["z"], data["params"]).astype(np.float32)
    with torch.no_grad():
        rhizome = np.concatenate([
            model(torch.from_numpy(cond[a:a + args.batch]).to(args.device),
                  torch.from_numpy(scal[a:a + args.batch]).to(args.device)).sigmoid()[:, 0].cpu().numpy()
            for a in range(0, n, args.batch)])
    print("predicted rhizome", flush=True)
    rlabel = f"Rhizome 2-D ({rhizome_dir.name})"
    preds = {rlabel: rhizome, label3d: data["pred3d"]}
    center = [j for j, (lo, hi) in enumerate(bands) if lo == 0 and hi == 1][0]
    for spec in args.fno2d:
        label, path = spec.split("=", 1)
        run = load_run(label, Path(path))
        preds[f"{label} 2-D"] = predict_fno(run, build_model, load_fno_checkpoint, norm_cls,
                                            data["bands"][:, center], data["z"], data["params"],
                                            args.device, args.batch)
        print(f"predicted {label}", flush=True)
    truth = data["truth"].astype(np.float64)
    names = list(preds)
    colors = dict(zip(names, COLORS))

    # Edge diagnostic (fno-21cm engine, slice mode), per-slice RMSE, spectra.
    cfg = BandConfig(mode="slice", threshold=0.5, dx_mpc=200.0 / 140, dy_mpc=200.0 / 140)
    by_cone = defaultdict(list)
    for i, c in enumerate(data["cone_id"]):
        by_cone[int(c)].append(i)
    edges = metrics.k_bins(truth.shape[-2:], 200.0 / 140)
    kc = 0.5 * (edges[1:] + edges[:-1])
    results, accums, summary = {}, {}, {}
    for name in names:
        pred = preds[name].astype(np.float64)
        acc = BoundaryBandAccumulator(cfg=cfg)
        for cone, idx in by_cone.items():
            acc.add_cone(cone, np.stack([pred[i] for i in idx], -1), np.stack([truth[i] for i in idx], -1))
        prof = acc.profile()
        d = np.asarray(prof["d_mpc"])
        with np.errstate(invalid="ignore", divide="ignore"):
            pab, paa, pbb = metrics.cross_power(pred, truth, 200.0 / 140, edges)
            ratio, r = np.nanmean(paa / pbb, 0), np.nanmean(pab / np.sqrt(paa * pbb), 0)
        high = kc > 0.5 * kc.max()
        results[name] = {"profile": prof, **acc.band_summary(),
                         "front_width_truth_mpc": front_width(d, np.asarray(prof["mean_truth"])),
                         "front_width_pred_mpc": front_width(d, np.asarray(prof["mean_pred"]))}
        accums[name] = acc
        summary[name] = {
            **{k: v for k, v in results[name].items() if k != "profile"},
            "slice_rmse_mean": float(np.sqrt(((pred - truth) ** 2).mean((1, 2))).mean()),
            "pooled_rmse": float(np.sqrt(((pred - truth) ** 2).mean())),
            "mean_xhi_mae": float(np.abs(pred.mean((1, 2)) - truth.mean((1, 2))).mean()),
            "power_ratio_high_k": float(np.nanmean(ratio[high])),
            "cross_correlation_high_k": float(np.nanmean(r[high])),
            "spectra": {"k": kc.tolist(), "ratio": ratio.tolist(), "r": r.tolist()},
        }
    per_cone = {nm: {rec["cone_id"]: rec for rec in accums[nm].per_cone} for nm in names}
    shared = sorted(set.intersection(*(set(p) for p in per_cone.values())))
    keys = [k for k in accums[names[0]].per_cone[0] if k.startswith(("L2_band", "H1_band"))]
    bootstrap = {other: {k: paired_bootstrap([per_cone[rlabel][c][k] for c in shared],
                                             [per_cone[other][c][k] for c in shared]) for k in keys}
                 for other in names[1:]}
    plot_overlay(results, cfg, out / "edge_band_overlay.png")
    (out / "summary.json").write_text(json.dumps(
        {"planes": n, "cones": len(by_cone), "fno3d_model": model3d, "manifest": str(Path(args.manifest).resolve()),
         "rhizome_checkpoint": {"run": str(rhizome_dir.resolve()), "step": checkpoint["step"]},
         "models": summary, "paired_bootstrap_rhizome_minus": bootstrap}, indent=2, default=float) + "\n")

    picks = representative(data["z"], truth.mean((1, 2)), [0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    info = [{"z": float(data["z"][i]), "cone_id": int(data["cone_id"][i]),
             "xhi_mean": float(truth[i].mean())} for i in picks]
    t_rep = truth[picks]
    field_grid(out / "representative_predictions.png", {"Truth": t_rep, **{k: preds[k][picks] for k in names}},
               ["Truth", *names], info, "Same native planes: rhizome 2-D vs fno-21cm 3-D (LOS windows) vs 2-D",
               "viridis", 0, 1, "$x_{HI}$", annotate=names, truth=t_rep)
    errors = {k: preds[k][picks] - t_rep for k in names}
    limit = max(0.05, float(np.quantile(np.abs(np.concatenate([e.ravel() for e in errors.values()])), 0.99)))
    field_grid(out / "representative_errors.png", errors, names, info, "Signed errors on the same planes",
               "RdBu_r", -limit, limit, "prediction - truth", annotate=names)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for name in names:
        s = summary[name]["spectra"]
        axes[0].plot(s["k"], s["ratio"], color=colors[name], label=name)
        axes[1].plot(s["k"], s["r"], color=colors[name], label=name)
    for ax, title in ((axes[0], "Power ratio $P_{pred}/P_{true}$"), (axes[1], "Cross-correlation $r(k)$")):
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("k [1/Mpc]")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(out / "spectra.png", dpi=160)
    plt.close(fig)

    width = max(len(k) for k in names)
    print(f"{'model':{width}s}  slice_rmse  pooled  L2_b2   H1_b2   L2_b5   H1_b5  front_w  Pk_hi  r_hi")
    for name in names:
        m = summary[name]
        print(f"{name:{width}s}  {m['slice_rmse_mean']:10.4f} {m['pooled_rmse']:7.4f} {m['L2_band2'][0]:6.4f} "
              f"{m['H1_band2'][0]:6.4f} {m['L2_band5'][0]:6.4f} {m['H1_band5'][0]:6.4f} "
              f"{m['front_width_pred_mpc']:7.2f} {m['power_ratio_high_k']:6.3f} {m['cross_correlation_high_k']:5.3f}")
    print(f"truth front width {summary[names[0]]['front_width_truth_mpc']:.2f} Mpc")
    for other, stats in bootstrap.items():
        print(f"rhizome - {other}: " + "  ".join(
            f"{k} {s['mean_diff']:+.4f} [{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}]" for k, s in stats.items()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
