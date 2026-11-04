"""Zero-shot evaluation of a 21cmFAST slice-wise rhizome on LoReLi II.

    python -m ebm21cm.evaluate_loreli --run runs/rhizome_mf2d_w64_structured_all \
        --data /pfs/10/work/hd_id260-fno_training/data/loreli_lc128 \
        --cache /pfs/10/work/hd_id260-fno_training/data/compressed/rhizome_mf_slices_all.h5 \
        --out runs/eval3d/loreli_lc128/rhizome_mf2d_w64_structured_all

For every valid slice of every cone the adapter builds the training inputs on
the 280-cell (400 Mpc) domain, the remapped checkpoint predicts x_HI and T_b,
the prediction is area-averaged back to the 128-cell LoReLi grid and
interpolated to the native redshifts, and the metrics below are accumulated
per stage (mean x_HI 0.05-0.35 / 0.35-0.65 / 0.65-0.95):

* x_HI RMSE at native resolution, after 2.3 and 4.7 Mpc Gaussian smoothing,
  and the full RMSE-vs-sigma curve (on a stratified subset);
* IoU at the truth's ionized fraction;
* transverse power ratio and r(k) for k <= 0.6 Mpc^-1;
* mean-free-path bubble-size W1 at x_HI < 0.1 / 0.5 / 0.9;
* 10-90 front width from the mean profile across the truth's x_HI = 0.5 contour;
* T_b RMSE and T_b PDF W1 (secondary);
* density-threshold and excursion-set baselines on the cone's own delta.

The border of the mirror-padded domain can be masked with ``--border`` cells
(the domain-check script measures how wide it has to be); the default is 0 and
the summary reports every metric on the remaining rectangular crop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch

from . import loreli_metrics as lm
from .data.loreli import LoreliAdapter, LoreliCone, loreli_files, read_cube, training_stats
from .data.loreli_splits import DEFAULT_SPLIT_FILE, load_split
from .loreli_inference import border_mask, downsample_transverse, load_model, map_to_native, predict_cone, predict_targets

SIGMAS = (0.0, 2.3, 4.7)
CURVE = (0.0, 0.5, 1.0, 1.5, 2.3, 3.0, 4.0, 4.7, 6.0, 8.0)
BUBBLE_THRESHOLDS = (0.1, 0.5, 0.9)
POWER_EDGES = lm.power_edges()


class Accumulator:
    """Per-stage sums; one instance per cone/stage and one merged over all cones."""

    def __init__(self):
        self.n = 0
        self.x_se = {s: 0.0 for s in SIGMAS}
        self.x_n = 0
        self.tb_se = 0.0
        self.tb_n = 0
        self.iou = 0.0
        self.iou_n = 0
        self.p_ab = np.zeros(len(POWER_EDGES) - 1)
        self.p_aa = np.zeros_like(self.p_ab)
        self.p_bb = np.zeros_like(self.p_ab)
        self.power_n = 0
        self.front_n = np.zeros(len(lm.FRONT_EDGES) - 1)
        self.front_true = np.zeros_like(self.front_n)
        self.front_pred = np.zeros_like(self.front_n)
        self.bubble = {t: [] for t in BUBBLE_THRESHOLDS}
        self.tb_w1 = []
        self.curve = {s: [0.0, 0] for s in CURVE}
        self.base = {name: {"iou": 0.0, "se": 0.0, "n": 0, "cells": 0} for name in ("density", "excursion")}

    def add_slice(self, pred, true, tb_pred, tb_true, cell):
        self.n += 1
        self.x_n += true.size
        self.x_se[0.0] += float(np.sum((pred - true) ** 2))
        for sigma in (2.3, 4.7):
            self.x_se[sigma] += float(np.sum((lm.smooth(pred, sigma, cell) - lm.smooth(true, sigma, cell)) ** 2))
        self.tb_se += float(np.sum((tb_pred - tb_true) ** 2))
        self.tb_n += tb_true.size
        iou = lm.iou_matched(pred, true)
        if np.isfinite(iou):
            self.iou += iou
            self.iou_n += 1
        p_ab, p_aa, p_bb = lm.binned_power(pred, true, cell, POWER_EDGES)
        if np.all(np.isfinite(p_ab)) and np.all(np.isfinite(p_aa)) and np.all(np.isfinite(p_bb)):
            self.p_ab += p_ab
            self.p_aa += p_aa
            self.p_bb += p_bb
            self.power_n += 1

    def add_shape(self, pred, true, tb_pred, tb_true, cell):
        self.tb_w1.append(lm.tb_pdf_w1(tb_pred, tb_true))
        for threshold in BUBBLE_THRESHOLDS:
            self.bubble[threshold].append(
                (lm.bubble_sizes(true, cell, threshold), lm.bubble_sizes(pred, cell, threshold)))
        profile = lm.front_profile(pred, true, cell, lm.FRONT_EDGES)
        if profile is not None:
            counts, sum_true, sum_pred = profile
            self.front_n += counts
            self.front_true += sum_true
            self.front_pred += sum_pred
        for sigma in CURVE:
            self.curve[sigma][0] += float(np.sum((lm.smooth(pred, sigma, cell) - lm.smooth(true, sigma, cell)) ** 2))
            self.curve[sigma][1] += true.size

    def add_baseline(self, name, pred, true):
        iou = lm.iou_matched(pred, true)
        entry = self.base[name]
        if np.isfinite(iou):
            entry["iou"] += iou
            entry["se"] += float(np.sum((pred - true) ** 2))
            entry["n"] += 1
            entry["cells"] += true.size

    def metrics(self):
        out = {"n_slices": self.n,
               "xhi_rmse": {str(s): float(np.sqrt(self.x_se[s] / max(self.x_n, 1))) for s in SIGMAS},
               "xhi_rmse_curve": {str(s): float(np.sqrt(self.curve[s][0] / max(self.curve[s][1], 1)))
                                  for s in CURVE},
               "tb_rmse_mk": float(np.sqrt(self.tb_se / max(self.tb_n, 1))),
               "iou_matched": float(self.iou / self.iou_n) if self.iou_n else float("nan"),
               "tb_pdf_w1": float(np.mean(self.tb_w1)) if self.tb_w1 else float("nan")}
        if self.power_n:
            ratio = self.p_aa / self.p_bb
            r = self.p_ab / np.sqrt(self.p_aa * self.p_bb)
            out["k_mpc"] = ((POWER_EDGES[:-1] + POWER_EDGES[1:]) / 2).tolist()
            out["power_ratio"] = ratio.tolist()
            out["power_r"] = r.tolist()
        for threshold in BUBBLE_THRESHOLDS:
            pairs = self.bubble[threshold]
            distances = [lm.metrics.size_distance(a, b) for a, b in pairs]
            distances = [x for x in distances if np.isfinite(x)]
            out[f"bubble_w1_{threshold}"] = float(np.mean(distances)) if distances else float("nan")
        if self.front_n.sum():
            d = (lm.FRONT_EDGES[:-1] + lm.FRONT_EDGES[1:]) / 2
            mean_true = np.divide(self.front_true, self.front_n, out=np.full_like(self.front_true, np.nan),
                                  where=self.front_n > 0)
            mean_pred = np.divide(self.front_pred, self.front_n, out=np.full_like(self.front_pred, np.nan),
                                  where=self.front_n > 0)
            out["front_width_true_mpc"] = lm.front_width(d, mean_true)
            out["front_width_pred_mpc"] = lm.front_width(d, mean_pred)
        for name, entry in self.base.items():
            out[f"baseline_{name}"] = {
                "iou_matched": float(entry["iou"] / entry["n"]) if entry["n"] else float("nan"),
                "xhi_rmse": float(np.sqrt(entry["se"] / entry["cells"])) if entry["cells"] else float("nan")}
        return out

    def merge(self, other):
        self.n += other.n
        for s in SIGMAS:
            self.x_se[s] += other.x_se[s]
        self.x_n += other.x_n
        self.tb_se += other.tb_se
        self.tb_n += other.tb_n
        self.iou += other.iou
        self.iou_n += other.iou_n
        self.p_ab += other.p_ab
        self.p_aa += other.p_aa
        self.p_bb += other.p_bb
        self.power_n += other.power_n
        self.front_n += other.front_n
        self.front_true += other.front_true
        self.front_pred += other.front_pred
        for t in BUBBLE_THRESHOLDS:
            self.bubble[t] += other.bubble[t]
        self.tb_w1 += other.tb_w1
        for s in CURVE:
            self.curve[s][0] += other.curve[s][0]
            self.curve[s][1] += other.curve[s][1]
        for name in self.base:
            for key in ("iou", "se", "n", "cells"):
                self.base[name][key] += other.base[name][key]


def select(items, limit):
    """Evenly spaced subset of at most ``limit`` items (all of them if limit <= 0)."""
    if limit <= 0 or len(items) <= limit:
        return list(items)
    return [items[i] for i in np.unique(np.linspace(0, len(items) - 1, limit).round().astype(int))]


def borders(keep):
    """Index border of a centred boolean square mask."""
    if keep.size == 0 or not keep.any():
        raise ValueError("empty transverse mask")
    row = np.flatnonzero(keep[keep.shape[0] // 2])[0]
    col = np.flatnonzero(keep[:, keep.shape[1] // 2])[0]
    return int(min(row, col))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_commit():
    try:
        root = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                        text=True, stderr=subprocess.DEVNULL).strip() != ""
        return commit, dirty
    except Exception:
        return "unknown", True


def fingerprint(args, best):
    """Complete config signature stored with each cone so a resume cannot mix runs."""
    commit, dirty = _code_commit()
    fp = {"run": str(args.run.resolve()), "step": best.get("step"),
          "checkpoint_sha256": _sha256(args.run / "best.pt"),
          "data": str(args.data.resolve()), "cache": str(args.cache.resolve()),
          "l_perp": args.l_perp, "border": args.border, "los_margin": args.los_margin,
          "slice_stride": args.slice_stride, "min_per_stage": args.min_per_stage,
          "shape_per_stage": args.shape_per_stage, "baseline_per_stage": args.baseline_per_stage,
          "no_baselines": bool(args.no_baselines),
          "density_mean": args.density_mean, "velocity": args.velocity,
          "device": str(torch.device(args.device)),
          "torch_inputs": bool(args.force_torch_inputs or torch.device(args.device).type != "cpu"),
          "n_native": args.n_native, "n_los": args.n_los,
          "z_range": list(args.z_range), "z_window": list(args.z_window) if args.z_window else None,
          "ids": None if not args.ids else sorted(int(i) for i in args.ids), "limit": args.limit,
          "code_commit": commit, "code_dirty": bool(dirty)}
    ids_from_split = getattr(args, "ids_from_split", None)
    if ids_from_split:
        fp["ids_from_split"] = ids_from_split
        fp["split_file"] = str(Path(args.split_file).resolve())
        fp["split_sha256"] = _sha256(args.split_file)
    return fp


def load_completed(out, fp):
    """(totals, records, done sims) from valid pickles; corrupt/partial pickles are removed for recompute."""
    totals = {name: Accumulator() for name, _, _ in lm.STAGES}
    records, done = [], set()
    for pkl in sorted(Path(out).glob("cone_*.pkl")):
        try:
            payload = pickle.loads(pkl.read_bytes())
            if not isinstance(payload, dict) or "fingerprint" not in payload or "stages" not in payload:
                raise ValueError("incomplete pickle")
        except Exception:
            pkl.unlink(missing_ok=True)
            continue
        if payload["fingerprint"] != fp:
            raise SystemExit(f"{pkl}: resume config differs from this command; use a fresh --out")
        for name in totals:
            totals[name].merge(payload["stages"][name])
        records.append(payload["record"])
        done.add(int(payload["record"]["sim"]))
    return totals, records, done


def select_targets(adapter, cone, truth_x, stride, min_per_stage):
    """Stage-stratified target indices: every ``stride``-th slice in each stage, at least ``min_per_stage``."""
    if stride <= 1:
        return None
    position = np.interp(adapter.target_distance, cone.distances, np.arange(cone.n_los))
    native = np.rint(position).astype(np.int64).clip(0, cone.n_los - 1)
    means = truth_x.reshape(-1, cone.n_los)[:, native].mean(0)
    stages = np.array([lm.stage_of(float(m)) for m in means])
    selected = []
    for name, _, _ in lm.STAGES:
        js = np.flatnonzero(stages == name)
        if len(js) == 0:
            continue
        count = min(len(js), max(int(min_per_stage), int(np.ceil(len(js) / stride))))
        pick = np.unique(np.linspace(0, len(js) - 1, count).round().astype(int))
        selected += js[pick].tolist()
    if not selected:
        raise ValueError("slice stride selected no active target slices")
    return np.sort(np.asarray(selected, dtype=np.int64))


def map_selected(adapter, cone, xhi, tb, targets, predicted):
    """Downsample selected predictions and assign each to the nearest native slice."""
    px = downsample_transverse(xhi, adapter, cone)
    pt = downsample_transverse(tb, adapter, cone)
    position = np.interp(adapter.target_distance, cone.distances, np.arange(cone.n_los))
    native = np.rint(position[targets]).astype(np.int64).clip(0, cone.n_los - 1)
    shape = (len(cone.z), cone.n_native, cone.n_native)
    out_x = np.full(shape, np.nan, np.float32)
    out_t = np.full(shape, np.nan, np.float32)
    for j, i in enumerate(native):
        if not cone.valid[i]:
            continue
        out_x[i], out_t[i] = px[j], pt[j]
        predicted[i] = True
    return out_x, out_t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="slice-wise mf2d run directory (best.pt)")
    ap.add_argument("--data", type=Path, required=True, help="LoReLi lc_128 folder")
    ap.add_argument("--cache", type=Path, required=True, help="training slice cache (bands + normalization)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ids", type=int, nargs="*", default=None, help="simulation numbers (default: all)")
    ap.add_argument("--ids-from-split", choices=("train", "val", "test"), default=None,
                    help="restrict to the simulation ids of this block split")
    ap.add_argument("--split-file", default=DEFAULT_SPLIT_FILE,
                    help="block-split JSON used by --ids-from-split")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--l-perp", type=float, default=294.0, choices=(294.0, 300.0))
    ap.add_argument("--density-mean", choices=("slice", "cone"), default="slice")
    ap.add_argument("--velocity", choices=("linear", "zero"), default="linear")
    ap.add_argument("--border", type=int, default=0, help="discard this many resampled cells at the domain edge")
    ap.add_argument("--los-margin", type=int, default=0,
                    help="exclude this many native slices at each valid end from the metrics (their LOS bands "
                         "are edge-replicated); predictions are still written")
    ap.add_argument("--slice-stride", type=int, default=1,
                    help="stage-stratified model stride: keep about 1/N of the active slices per stage (1 = all)")
    ap.add_argument("--min-per-stage", type=int, default=8,
                    help="keep at least this many slices per stage when --slice-stride > 1")
    ap.add_argument("--resume", action="store_true", help="skip cones whose per-cone pickle is already present")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force-torch-inputs", action="store_true",
                    help="debug: run the torch input path even on CPU (it is implied by --device cuda)")
    ap.add_argument("--shape-per-stage", type=int, default=0,
                    help="slices per stage for bubble/front/PDF/curve metrics; 0 = all active slices (production)")
    ap.add_argument("--baseline-per-stage", type=int, default=32,
                    help="slices per stage for the excursion-set baseline (0 = all shape slices)")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--save-cones", action="store_true", help="write x_HI/T_b predictions per cone as float16 h5")
    ap.add_argument("--n-native", type=int, default=128, help="debug/synthetic: native transverse size")
    ap.add_argument("--n-los", type=int, default=1536, help="debug/synthetic: native LOS length")
    ap.add_argument("--z-range", type=float, nargs=2, default=(5.0, 15.0), help="debug/synthetic: LOS redshift range")
    ap.add_argument("--z-window", type=float, nargs=2, default=None,
                    help="evaluate only this redshift window (e.g. 8.5 8.7 for a quick real-data check)")
    args = ap.parse_args(argv)
    if args.slice_stride < 1 or args.min_per_stage < 1:
        ap.error("--slice-stride and --min-per-stage must be >= 1")
    args.out.mkdir(parents=True, exist_ok=True)
    stats = training_stats(args.cache)
    native, model, best = load_model(args.run)
    device = torch.device(args.device)
    model = model.to(device)
    files, sims = loreli_files(args.data)
    if args.ids_from_split:
        split = load_split(args.split_file)
        wanted = set(split[f"{args.ids_from_split}_sims"])
        files = [p for p, s in zip(files, sims) if int(s) in wanted]
        sims = [s for s in sims if int(s) in wanted]
    if args.ids:
        wanted = set(args.ids)
        files = [p for p, s in zip(files, sims) if s in wanted]
        sims = [s for s in sims if s in wanted]
    if args.limit:
        files = files[:args.limit]
        sims = sims[:args.limit]
    if not files:
        raise SystemExit("no cones selected")
    fp = fingerprint(args, best)

    totals = {name: Accumulator() for name, _, _ in lm.STAGES}
    per_cone = []
    if args.resume:
        totals, per_cone, done = load_completed(args.out, fp)
        files = [p for p in files if int(p.stem.split("simu")[-1]) not in done]
        print(f"resume: {len(per_cone)} cones already done, {len(files)} to go", flush=True)
    print(f"{len(files)} cones to evaluate, {len(stats['bands'])} bands, checkpoint step {best.get('step')}; "
          f"stride {args.slice_stride} (min {args.min_per_stage}/stage), LOS margin {args.los_margin}, "
          f"device {device}; scalars = oracle histories + training-mean parameters", flush=True)

    def load(path):
        return read_cube(path)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(load, files[0]) if files else None
        for k, path in enumerate(files):
            sim = int(path.stem.split("simu")[-1])
            target = args.out / f"cone_{sim}.json"
            started = time.time()
            data = future.result()
            if k + 1 < len(files):
                future = pool.submit(load, files[k + 1])
            cone = LoreliCone(path, l_perp=args.l_perp, density_mean=args.density_mean, n_native=args.n_native,
                              n_los=args.n_los, z_range=tuple(args.z_range), data=data)
            if args.z_window is not None:
                cone.valid &= (cone.z >= args.z_window[0]) & (cone.z <= args.z_window[1])
                cone.n_slices = int(cone.valid.sum())
                if not cone.n_slices:
                    raise SystemExit(f"{path}: no valid slices in z window {args.z_window}")
            adapter = LoreliAdapter(cone, stats, velocity=args.velocity, device=device,
                                    force_torch=args.force_torch_inputs)
            raw = cone.read(0, cone.n_los)
            truth_x = 1.0 - raw["xion"]
            truth_tb = raw["dtb"]
            delta = cone.delta(raw["dens"])
            targets = select_targets(adapter, cone, truth_x, args.slice_stride, args.min_per_stage)
            predicted = np.zeros(cone.n_los, bool)
            if targets is None:
                xhi, tb = predict_cone(adapter, model, best, device, args.batch_size)
                pred_x, pred_tb, keep = map_to_native(xhi, tb, adapter, cone, args.border)
                predicted[cone.valid] = True
            else:
                xhi, tb, targets = predict_targets(adapter, model, best, device, args.batch_size, targets)
                pred_x, pred_tb = map_selected(adapter, cone, xhi, tb, targets, predicted)
                keep = None
            if keep is None:
                keep = border_mask(adapter, cone, args.border)
            border = borders(keep)
            crop = slice(border, cone.n_native - border)
            stages = {name: Accumulator() for name, _, _ in lm.STAGES}
            active = {name: [] for name in stages}
            valid = np.flatnonzero(cone.valid)
            edge = set(valid[:args.los_margin]) | set(valid[-args.los_margin:]) if args.los_margin else set()
            for i in valid:
                if i in edge or not predicted[i]:
                    continue
                true = truth_x[crop, crop, i]
                stage = lm.stage_of(float(true.mean()))
                if stage is None:
                    continue
                active[stage].append(int(i))
                stages[stage].add_slice(pred_x[i, crop, crop], true, pred_tb[i, crop, crop],
                                        truth_tb[crop, crop, i], cone.l_perp / cone.n_native)
            for name, idx in active.items():
                shape_idx = idx if args.shape_per_stage <= 0 else select(idx, args.shape_per_stage)
                for i in shape_idx:
                    true = truth_x[crop, crop, i]
                    stages[name].add_shape(pred_x[i, crop, crop], true, pred_tb[i, crop, crop],
                                           truth_tb[crop, crop, i], cone.l_perp / cone.n_native)
                if not args.no_baselines:
                    delta_slice = delta[crop, crop]
                    for i in select(idx, args.baseline_per_stage or args.shape_per_stage):
                        true = truth_x[crop, crop, i]
                        score = delta_slice[:, :, i]
                        fraction = float((true < 0.5).mean())
                        if not 0 < fraction < 1:
                            continue
                        threshold = np.quantile(score, 1 - fraction)
                        stages[name].add_baseline("density", np.where(score >= threshold, 0.0, 1.0), true)
                        es = lm.sharp_k_max(score, cone.l_perp / cone.n_native)
                        threshold = np.quantile(es, 1 - fraction)
                        stages[name].add_baseline("excursion", np.where(es >= threshold, 0.0, 1.0), true)
            record = {"sim": sim, "path": str(path), "l_perp": args.l_perp, "density_mean": args.density_mean,
                      "velocity": args.velocity, "border": args.border, "los_margin": args.los_margin,
                      "slice_stride": args.slice_stride, "min_per_stage": args.min_per_stage,
                      "n_valid": int(cone.n_slices), "n_predicted": int(predicted.sum()),
                      "n_edge_excluded": len(edge), "n_targets": None if targets is None else int(len(targets)),
                      "shape_per_stage": args.shape_per_stage, "baseline_per_stage": args.baseline_per_stage,
                      "no_baselines": bool(args.no_baselines),
                      "z_window": list(args.z_window) if args.z_window else None,
                      "ids_from_split": args.ids_from_split,
                      "split_sha256": fp.get("split_sha256"),
                      "device": str(device), "torch_inputs": bool(adapter.torch_inputs),
                      "code_commit": fp["code_commit"], "code_dirty": fp["code_dirty"],
                      "scalars": {"histories": "oracle (cone's own x_HI/T_b band means)",
                                  "parameters": "training mean (normalized zero)"},
                      "n_active": {name: len(idx) for name, idx in active.items()},
                      "stages": {name: stages[name].metrics() for name in stages},
                      "seconds": round(time.time() - started, 1)}
            if args.save_cones:
                with h5py.File(args.out / f"cone_{sim}.h5", "w") as f:
                    f.create_dataset("xhi", data=pred_x.astype(np.float16), compression="gzip", compression_opts=4)
                    f.create_dataset("tb", data=pred_tb.astype(np.float16), compression="gzip", compression_opts=4)
                    f.create_dataset("valid", data=cone.valid)
                    f.create_dataset("predicted", data=predicted)
                    f.create_dataset("z", data=cone.z)
                    if targets is not None:
                        f.create_dataset("target_index", data=targets)
                    f.attrs.update({"sim": sim, "checkpoint": str((args.run / "best.pt").resolve()),
                                    "border": args.border, "los_margin": args.los_margin,
                                    "slice_stride": args.slice_stride, "grid": "native 128, increasing z"})
            target.write_text(json.dumps(record, indent=1))
            tmp = args.out / f".cone_{sim}.pkl.tmp"
            tmp.write_bytes(pickle.dumps({"fingerprint": fp, "record": record, "stages": stages}))
            os.replace(tmp, args.out / f"cone_{sim}.pkl")
            per_cone.append(record)
            for name in totals:
                totals[name].merge(stages[name])
            print(f"[{k + 1}/{len(files)}] {sim}: {record['seconds']} s, "
                  + "  ".join(f"{name} n={record['n_active'][name]} "
                              f"rmse={record['stages'][name]['xhi_rmse']['0.0']:.4f} "
                              f"iou={record['stages'][name]['iou_matched']:.3f}" for name in stages), flush=True)

    stages = {}
    for name in totals:
        entry = totals[name].metrics()
        entry.pop("front_width_pred_mpc", None)
        entry.pop("front_width_true_mpc", None)
        entry.update(front_stats(per_cone, name))
        stages[name] = entry
    summary = {"run": str(args.run.resolve()), "data": str(args.data.resolve()), "cache": str(args.cache.resolve()),
               "checkpoint_step": best.get("step"), "l_perp": args.l_perp, "density_mean": args.density_mean,
               "velocity": args.velocity, "border": args.border, "los_margin": args.los_margin,
               "slice_stride": args.slice_stride, "min_per_stage": args.min_per_stage,
               "shape_per_stage": args.shape_per_stage, "baseline_per_stage": args.baseline_per_stage,
               "no_baselines": bool(args.no_baselines),
               "device": str(device), "torch_inputs": bool(args.force_torch_inputs or device.type != "cpu"),
               "ids": None if not args.ids else sorted(int(i) for i in args.ids), "limit": args.limit,
               "ids_from_split": args.ids_from_split, "split_sha256": fp.get("split_sha256"),
               "z_window": list(args.z_window) if args.z_window else None,
               "n_cones": len(per_cone),
               "scalars": {"histories": "oracle (cone's own x_HI/T_b band means)",
                           "parameters": "training mean (normalized zero)"},
               "stages": stages}
    (args.out / "summary.json").write_text(json.dumps({"summary": summary, "cones": per_cone}, indent=1))
    write_markdown(args.out / "summary.md", summary)
    print(f"wrote {args.out / 'summary.json'}", flush=True)


def front_stats(records, stage):
    """Median finite front width per stage over cones, plus the number of NaN/missing cones (never imputed)."""
    pred = [r["stages"].get(stage, {}).get("front_width_pred_mpc", float("nan")) for r in records]
    true = [r["stages"].get(stage, {}).get("front_width_true_mpc", float("nan")) for r in records]
    def median(values):
        finite = [float(v) for v in values if np.isfinite(v)]
        return float(np.median(finite)) if finite else float("nan")
    return {"front_width_pred_mpc_median": median(pred),
            "front_width_true_mpc_median": median(true),
            "front_width_pred_nan_cones": int(sum(not np.isfinite(v) for v in pred)),
            "front_width_true_nan_cones": int(sum(not np.isfinite(v) for v in true)),
            "front_width_n_cones": len(records)}


def write_markdown(path, summary):
    lines = [f"# LoReLi zero-shot: {Path(summary['run']).name}", "",
             f"{summary['n_cones']} cones, L_perp = {summary['l_perp']:.0f} Mpc, "
             f"density mean = {summary['density_mean']}, velocity = {summary['velocity']}, border = {summary['border']}, "
             f"LOS margin = {summary['los_margin']}, slice stride = {summary['slice_stride']} "
             f"(min {summary['min_per_stage']}/stage)",
             "",
             "Scalars are the cone's own (oracle) histories and the training-mean parameters; the numbers are therefore "
             "a history-conditioned upper bound, not a parameter-marginalized score. The first/last `--los-margin` "
             "slices use edge-replicated LOS bands.",
             f"shape slices = {'all active' if summary['shape_per_stage'] <= 0 else summary['shape_per_stage']} per stage, "
             f"baselines = {summary['baseline_per_stage']} per stage",
             "", "| stage | n | x_HI RMSE | 2.3 Mpc | 4.7 Mpc | IoU | T_b RMSE [mK] | T_b PDF W1 | bubble W1 (0.5) | front width pred/truth [Mpc] (NaN cones) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, _, _ in lm.STAGES:
        m = summary["stages"][name]
        lines.append(f"| {name} | {m['n_slices']} | {m['xhi_rmse']['0.0']:.4f} | {m['xhi_rmse']['2.3']:.4f} | "
                     f"{m['xhi_rmse']['4.7']:.4f} | {m['iou_matched']:.3f} | {m['tb_rmse_mk']:.2f} | "
                     f"{m['tb_pdf_w1']:.3f} | {m['bubble_w1_0.5']:.3f} | "
                     f"{m.get('front_width_pred_mpc_median', float('nan')):.2f} / "
                     f"{m.get('front_width_true_mpc_median', float('nan')):.2f} "
                     f"({m.get('front_width_pred_nan_cones', 0)}) |")
    lines += ["", "| stage | density IoU | density RMSE | excursion-set IoU | excursion-set RMSE |",
              "|---|---|---|---|---|"]
    for name, _, _ in lm.STAGES:
        b = summary["stages"][name]
        d, e = b["baseline_density"], b["baseline_excursion"]
        lines.append(f"| {name} | {d['iou_matched']:.3f} | {d['xhi_rmse']:.3f} | "
                     f"{e['iou_matched']:.3f} | {e['xhi_rmse']:.3f} |")
    lines += ["", "At late stages the model IoU is close to the density baseline; use the 4.7 Mpc RMSE and the "
                  "cross-correlation there, not IoU, for morphology."]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
