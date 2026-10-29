"""CPU diagnostic: how much of the LoReLi zero-shot x_HI error is output shape?

HAR-63, follow-up of HAR-39.  No training, no GPU, predictions read-only.

For each reionization stage (by true slice-mean x_HI) a monotone map is fitted
on the *training blocks* of ``runs/loreli_splits/block_split_v2_seed0.json``:

* ``isotonic``: ``sklearn.isotonic.IsotonicRegression`` (increasing, clipped to
  [0, 1]) on up to ~2M train-block cells per stage;
* ``logistic``: two parameters ``(a, b)`` with ``x' = sigmoid(a * logit(x) + b)``,
  fitted by least squares on the same cells.

Held-out blocks are then scored with exactly the slice/crop selection of
``ebm21cm.evaluate_loreli`` (border 4 resampled cells -> a 124-cell crop, LOS
margin 28, stage from the true slice mean), so the "before" column is the same
quantity that :mod:`ebm21cm.evaluate_loreli` reports.  Predictions come from
the saved float16 ``cone_*.h5`` files, so values differ from the float32 run by
float16 rounding; the script quantifies that against the per-cone records.

The corrected IoU selects the ``n`` lowest-x_HI cells, so a monotone map cannot
change it.  Two numbers are reported: the tie-aware mask (ties in the mapped
field are broken by the pre-map value, i.e. the map is applied as a relabelling)
and the naive mask computed directly on the mapped float32 field, where equal
values are broken by cell index.  The tie-aware IoU is asserted to equal the
"before" value; any naive deviation is a float tie artefact and is reported.

    python scripts/loreli_monotone_map.py \
        --pred runs/eval3d/loreli_lc128/rhizome_mf2d_w64_structured_all_full_v2 \
        --data /pfs/10/work/hd_id260-fno_training/data/loreli_lc128 \
        --split runs/loreli_splits/block_split_v2_seed0.json \
        --out runs/eval3d/loreli_lc128/monotone_map_v2
"""

from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
from scipy.optimize import least_squares

from ebm21cm import loreli_metrics as lm
from ebm21cm.data.loreli import LoreliCone, TRAINING_CELL_MPC, loreli_files, read_cube
from ebm21cm.data.loreli_splits import DEFAULT_SPLIT_FILE
from ebm21cm.evaluate_loreli import borders
from ebm21cm.loreli_inference import border_mask

SIGMAS = (0.0, 2.3, 4.7)
FIELDS = ("before", "isotonic", "logistic")
MAPS = ("isotonic", "logistic")
# Matched degraded-21cmFAST floor (runs/eval3d/loreli_lc128/degraded_matched, 199 cones).
# The file stores the local column twice (2.3 == 4.7); native and 4.7 are reported.
FLOOR = {
    "xHI_0.05-0.35": {"0.0": 0.1932, "4.7": 0.1458},
    "xHI_0.35-0.65": {"0.0": 0.1844, "4.7": 0.1230},
    "xHI_0.65-0.95": {"0.0": 0.0998, "4.7": 0.0506},
}
N_FIT = 2_000_000
PER_SLICE = 800
CLIP = 1e-6
BORDER = 4
LOS_MARGIN = 28
L_PERP = 294.0
N_NATIVE = 128
POWER_EDGES = lm.power_edges()
K_MPC = ((POWER_EDGES[:-1] + POWER_EDGES[1:]) / 2).tolist()


def sigmoid(t):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(t, dtype=np.float64)))


def logit(field):
    clipped = np.clip(field, CLIP, 1.0 - CLIP)
    return np.log(clipped / (1.0 - clipped))


def logistic_apply(field, a, b):
    return np.clip(sigmoid(a * logit(field) + b), 0.0, 1.0).astype(np.float32)


def fit_logistic(pred, true):
    """Least-squares (a, b) in x' = sigmoid(a * logit(x) + b); a stays positive."""
    u = logit(pred)

    def residual(theta):
        return sigmoid(theta[0] * u + theta[1]) - true

    result = least_squares(residual, np.array([1.0, 0.0]), bounds=([1e-4, -50.0], [100.0, 50.0]),
                           method="trf", max_nfev=200)
    return float(result.x[0]), float(result.x[1]), float(np.sqrt(np.mean(result.fun ** 2)))


def iou_ranked(pred, true, tie=None, threshold=0.5):
    """IoU of the ``n = count(true < threshold)`` lowest-x_HI cells.

    ``tie`` breaks equal values in ``pred`` (ascending, as the pre-map value
    does); with ``tie=None`` this reproduces :func:`ebm21cm.loreli_metrics.iou_matched`.
    """
    ion = np.asarray(true) < threshold
    n = int(np.count_nonzero(ion))
    if n == 0 or n == ion.size:
        return float("nan")
    flat = np.asarray(pred, dtype=np.float64).ravel()
    order = np.argsort(flat, kind="stable") if tie is None else np.lexsort((np.asarray(tie).ravel(), flat))
    guess = np.zeros(ion.size, dtype=bool)
    guess[order[:n]] = True
    union = np.count_nonzero(ion.ravel() | guess)
    return float(np.count_nonzero(ion.ravel() & guess) / union) if union else float("nan")


class Score:
    """Per-stage sums over slices."""

    def __init__(self):
        self.n_slices = 0
        self.cells = 0
        self.se = {f: {s: 0.0 for s in SIGMAS} for f in FIELDS}
        self.iou = {f: 0.0 for f in FIELDS}
        self.iou_tied = {f: 0.0 for f in FIELDS}
        self.iou_n = {f: 0 for f in FIELDS}
        self.iou_delta_max = {f: 0.0 for f in MAPS}
        self.iou_delta_tied_max = {f: 0.0 for f in MAPS}
        self.iou_worst = {}
        self.partial = {f: 0.0 for f in ("truth",) + FIELDS}
        self.p_ab = {f: np.zeros(len(POWER_EDGES) - 1) for f in FIELDS}
        self.p_aa = {f: np.zeros(len(POWER_EDGES) - 1) for f in FIELDS}
        self.p_bb = {f: np.zeros(len(POWER_EDGES) - 1) for f in FIELDS}
        self.power_n = 0

    def add(self, planes, true, cell, meta=None):
        self.n_slices += 1
        self.cells += true.size
        self.partial["truth"] += float(np.count_nonzero((true > 0.05) & (true < 0.95)))
        for s in SIGMAS:
            smooth_true = lm.smooth(true, s, cell)
            for f in FIELDS:
                self.se[f][s] += float(np.sum((lm.smooth(planes[f], s, cell) - smooth_true) ** 2))
        for f in FIELDS:
            self.partial[f] += float(np.count_nonzero((planes[f] > 0.05) & (planes[f] < 0.95)))
            value = iou_ranked(planes[f], true)
            if np.isfinite(value):
                self.iou[f] += value
                self.iou_n[f] += 1
            if f == "before":
                self._base = value
                self._base_tied = value
            else:
                tied = iou_ranked(planes[f], true, tie=planes["before"])
                if np.isfinite(value) and np.isfinite(self._base):
                    delta = abs(value - self._base)
                    self.iou_delta_max[f] = max(self.iou_delta_max[f], delta)
                    if delta >= self.iou_delta_max[f] and meta is not None:
                        self.iou_worst[f] = dict(meta, iou_before=self._base, iou_direct=value, delta=delta,
                                                 n_ionized=int(np.count_nonzero(true < 0.5)),
                                                 n_cells=int(true.size),
                                                 distinct_before=int(np.unique(planes["before"]).size),
                                                 distinct_after=int(np.unique(planes[f]).size))
                if np.isfinite(tied) and np.isfinite(self._base_tied):
                    self.iou_tied[f] += tied
                    self.iou_delta_tied_max[f] = max(self.iou_delta_tied_max[f], abs(tied - self._base_tied))
            p_ab, p_aa, p_bb = lm.binned_power(planes[f], true, cell, POWER_EDGES)
            if np.all(np.isfinite(p_ab)) and np.all(np.isfinite(p_aa)) and np.all(np.isfinite(p_bb)):
                self.p_ab[f] += p_ab
                self.p_aa[f] += p_aa
                self.p_bb[f] += p_bb
        self.power_n += 1

    def metrics(self, stage):
        out = {"n_slices": self.n_slices, "n_cells": self.cells, "xhi_rmse": {}, "iou": {}, "iou_tied": {},
               "iou_max_abs_delta_vs_before": {}, "iou_max_abs_delta_tied_vs_before": {},
               "iou_worst_case": self.iou_worst, "partial_fraction": {}, "power": {}}
        for f in FIELDS:
            out["xhi_rmse"][f] = {str(s): float(np.sqrt(self.se[f][s] / max(self.cells, 1))) for s in SIGMAS}
            out["iou"][f] = float(self.iou[f] / self.iou_n[f]) if self.iou_n[f] else float("nan")
            out["partial_fraction"][f] = self.partial[f] / max(self.cells, 1)
        out["iou_tied"]["before"] = out["iou"]["before"]
        for f in MAPS:
            out["iou_tied"][f] = (float(self.iou_tied[f] / self.iou_n["before"])
                                  if self.iou_n["before"] else float("nan"))
            out["iou_max_abs_delta_vs_before"][f] = self.iou_delta_max[f]
            out["iou_max_abs_delta_tied_vs_before"][f] = self.iou_delta_tied_max[f]
        out["partial_fraction"]["truth"] = self.partial["truth"] / max(self.cells, 1)
        if self.power_n:
            out["power"]["k_mpc"] = K_MPC
            out["power"]["ratio"] = {}
            out["power"]["ratio_max_le_0.6"] = {}
            out["power"]["ratio_mean"] = {}
            for f in FIELDS:
                ratio = np.divide(self.p_aa[f], self.p_bb[f], out=np.full_like(self.p_aa[f], np.nan),
                                  where=self.p_bb[f] > 0)
                out["power"]["ratio"][f] = ratio.tolist()
                finite = ratio[np.isfinite(ratio)]
                out["power"]["ratio_max_le_0.6"][f] = float(np.max(finite)) if finite.size else float("nan")
                out["power"]["ratio_mean"][f] = float(np.mean(finite)) if finite.size else float("nan")
        floor = FLOOR[stage]
        out["excess_pct_vs_floor"] = {
            f: {key: float(100.0 * (out["xhi_rmse"][f][key] / floor[key] - 1.0)) for key in ("0.0", "4.7")}
            for f in FIELDS}
        return out


def selection_mask(cone, n_out):
    """Border crop exactly as ``evaluate_loreli`` computes it for ``--border 4``."""
    keep = border_mask(types.SimpleNamespace(n_out=n_out), cone, BORDER)
    return keep, borders(keep)


def active_slices(cone, predicted):
    """Valid, predicted, non-edge slices in ``evaluate_loreli`` order."""
    valid = np.flatnonzero(cone.valid)
    edge = set(valid[:LOS_MARGIN]) | set(valid[-LOS_MARGIN:]) if LOS_MARGIN else set()
    return [int(i) for i in valid if i not in edge and predicted[i]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=Path(DEFAULT_SPLIT_FILE))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fit-cells", type=int, default=N_FIT)
    ap.add_argument("--per-slice", type=int, default=PER_SLICE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", type=int, nargs="*", default=None, help="debug: restrict to these sims")
    args = ap.parse_args(argv)

    from sklearn.isotonic import IsotonicRegression

    split = json.loads(args.split.read_text())
    role = {}
    for name in ("train", "val", "test"):
        for sim in split["splits"][name]["sims"]:
            role[int(sim)] = name
    raw_files, raw_sims = loreli_files(args.data)
    pairs = [(path, sim) for path, sim in zip(raw_files, raw_sims)
             if sim in role and (args.pred / f"cone_{sim}.h5").exists()
             and (args.ids is None or sim in set(args.ids))]
    if not pairs:
        raise SystemExit("no cones with both a raw file and a saved prediction")
    print(f"{len(pairs)} cones: " + ", ".join(
        f"{r} {sum(1 for _, s in pairs if role[s] == r)}" for r in ("train", "val", "test")), flush=True)

    n_out = int(round(L_PERP / TRAINING_CELL_MPC))
    probe = types.SimpleNamespace(n_native=N_NATIVE, l_perp=L_PERP)
    keep = border_mask(types.SimpleNamespace(n_out=n_out), probe, BORDER)
    border = borders(keep)
    cells = int(keep.sum())
    print(f"crop: border {BORDER} resampled cells -> index {border}, {N_NATIVE - 2 * border} cells, "
          f"{cells} per slice", flush=True)

    # ---------------- phase 1: fit on train blocks ----------------
    rng = np.random.default_rng(args.seed)
    sampled = {name: {"pred": [], "true": [], "n_slices": 0, "cones": set()}
               for name, _, _ in lm.STAGES}
    started = time.time()
    done = 0
    for path, sim in pairs:
        if role[sim] != "train":
            continue
        with h5py.File(args.pred / f"cone_{sim}.h5", "r") as handle:
            pred_x = handle["xhi"][:].astype(np.float32)
            predicted = handle["predicted"][:]
        cone = LoreliCone(path, l_perp=L_PERP, data=read_cube(path))
        truth_x = 1.0 - cone.read(0, cone.n_los, fields=("xion",))["xion"]
        for i in active_slices(cone, predicted):
            true = truth_x[border:N_NATIVE - border, border:N_NATIVE - border, i]
            stage = lm.stage_of(float(true.mean()))
            if stage is None:
                continue
            plane = pred_x[i, border:N_NATIVE - border, border:N_NATIVE - border]
            pick = rng.integers(0, true.size, args.per_slice)
            sampled[stage]["pred"].append(plane.ravel()[pick])
            sampled[stage]["true"].append(true.ravel()[pick])
            sampled[stage]["n_slices"] += 1
            sampled[stage]["cones"].add(sim)
        done += 1
        if done % 20 == 0:
            print(f"  fit pass: {done} train cones, {time.time() - started:.0f} s", flush=True)

    fits = {}
    models = {}
    for stage, _, _ in lm.STAGES:
        pred = np.concatenate(sampled[stage]["pred"]) if sampled[stage]["pred"] else np.zeros(0, np.float32)
        true = np.concatenate(sampled[stage]["true"]) if sampled[stage]["true"] else np.zeros(0, np.float32)
        if pred.size == 0:
            fits[stage] = {"n_cells": 0}
            print(f"{stage}: no training cells", flush=True)
            continue
        if pred.size > args.fit_cells:
            pick = np.random.default_rng(args.seed + 1).choice(pred.size, args.fit_cells, replace=False)
            pred, true = pred[pick], true[pick]
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(pred.astype(np.float64), true.astype(np.float64))
        a, b, rmse_logistic = fit_logistic(pred, true)
        models[stage] = iso
        fits[stage] = {
            "n_cells": int(pred.size),
            "n_slices": sampled[stage]["n_slices"],
            "n_cones": len(sampled[stage]["cones"]),
            "isotonic": {"x": np.asarray(iso.X_thresholds_, float).tolist(),
                         "y": np.asarray(iso.y_thresholds_, float).tolist(),
                         "n_knots": int(len(iso.X_thresholds_))},
            "logistic": {"a": a, "b": b, "train_rmse": rmse_logistic},
        }
        print(f"{stage}: {pred.size} cells / {sampled[stage]['n_slices']} slices / "
              f"{fits[stage]['n_cones']} cones; isotonic {fits[stage]['isotonic']['n_knots']} knots; "
              f"logistic a={a:.4f} b={b:.4f}", flush=True)

    # ---------------- phase 2: score val and test blocks ----------------
    scores = {name: {stage: Score() for stage, _, _ in lm.STAGES} for name in ("val", "test")}
    per_cone, checked = [], []
    for path, sim in pairs:
        if role[sim] == "train":
            continue
        with h5py.File(args.pred / f"cone_{sim}.h5", "r") as handle:
            pred_x = handle["xhi"][:].astype(np.float32)
            predicted = handle["predicted"][:]
            saved_valid = handle["valid"][:]
        cone = LoreliCone(path, l_perp=L_PERP, data=read_cube(path))
        if not np.array_equal(saved_valid, cone.valid):
            raise SystemExit(f"{path}: saved valid mask differs from the raw cube")
        truth_x = 1.0 - cone.read(0, cone.n_los, fields=("xion",))["xion"]
        cell = cone.l_perp / cone.n_native
        cone_score = {stage: Score() for stage, _, _ in lm.STAGES}
        stage_seen = set()
        for i in active_slices(cone, predicted):
            true = truth_x[border:N_NATIVE - border, border:N_NATIVE - border, i]
            stage = lm.stage_of(float(true.mean()))
            if stage is None or stage not in models:
                continue
            stage_seen.add(stage)
            plane = pred_x[i, border:N_NATIVE - border, border:N_NATIVE - border]
            if not np.all(np.isfinite(plane)):
                raise SystemExit(f"{path}: non-finite prediction in slice {i}")
            planes = {
                "before": plane,
                "isotonic": np.clip(models[stage].predict(plane.ravel().astype(np.float64)), 0.0, 1.0)
                            .reshape(plane.shape).astype(np.float32),
                "logistic": logistic_apply(plane, fits[stage]["logistic"]["a"], fits[stage]["logistic"]["b"]),
            }
            meta = {"sim": sim, "slice": int(i), "z": float(cone.z[i])}
            for target in (scores[role[sim]][stage], cone_score[stage]):
                target.add(planes, true, cell, meta)
        record = json.loads((args.pred / f"cone_{sim}.json").read_text())
        for stage in sorted(stage_seen):
            m = cone_score[stage].metrics(stage)
            r = record["stages"][stage]
            checked.append({"sim": sim, "stage": stage,
                            "n_slices_mine": m["n_slices"], "n_slices_record": r["n_slices"],
                            "rmse_mine": m["xhi_rmse"]["before"]["0.0"], "rmse_record": r["xhi_rmse"]["0.0"],
                            "iou_mine": m["iou"]["before"], "iou_record": r["iou_matched"]})
        per_cone.append({"sim": sim, "split": role[sim], "stages": sorted(stage_seen)})

    # ---------------- assemble ----------------
    result = {"pred": str(args.pred), "data": str(args.data), "split": str(args.split),
              "border_resampled_cells": BORDER, "crop_border_index": border,
              "crop_cells_per_side": N_NATIVE - 2 * border, "cells_per_slice": cells,
              "los_margin": LOS_MARGIN, "l_perp": L_PERP, "n_out": n_out,
              "fit_cells_requested": args.fit_cells, "per_slice_cells": args.per_slice, "seed": args.seed,
              "floor": FLOOR, "n_blocks": split["n_blocks"], "n_blocks_used": split["n_blocks_used"],
              "split_cones": {k: split["splits"][k]["n_cones"] for k in ("train", "val", "test")},
              "split_blocks": {k: split["splits"][k]["n_blocks"] for k in ("train", "val", "test")},
              "fit": fits, "scores": {}, "per_cone": per_cone, "selection_check": {}}
    for name in ("val", "test"):
        result["scores"][name] = {}
        for stage, _, _ in lm.STAGES:
            entry = scores[name][stage].metrics(stage)
            entry["n_cones"] = sum(1 for c in per_cone if c["split"] == name and stage in c["stages"])
            result["scores"][name][stage] = entry

    if checked:
        n_bad = sum(1 for c in checked if c["n_slices_mine"] != c["n_slices_record"])
        rmse_d = max(abs(c["rmse_mine"] - c["rmse_record"]) for c in checked)
        iou_d = max(abs(c["iou_mine"] - c["iou_record"]) for c in checked)
    else:
        n_bad, rmse_d, iou_d = 0, float("nan"), float("nan")
    result["selection_check"] = {
        "n_cone_stages": len(checked), "slice_count_mismatches": n_bad,
        "max_abs_rmse_diff_vs_record": rmse_d, "max_abs_iou_diff_vs_record": iou_d,
        "note": "records were computed from float32 predictions; these from the saved float16 files",
        "rows": checked,
    }
    result["iou_assertion"] = {
        name: {stage: {f: {key: result["scores"][name][stage][key][f] for key in
                           ("iou_max_abs_delta_tied_vs_before", "iou_max_abs_delta_vs_before")}
                       for f in MAPS}
               for stage, _, _ in lm.STAGES} for name in ("val", "test")}
    tied = [inner["iou_max_abs_delta_tied_vs_before"] for name in result["iou_assertion"].values()
            for stage in name.values() for inner in stage.values()]
    naive = [inner["iou_max_abs_delta_vs_before"] for name in result["iou_assertion"].values()
             for stage in name.values() for inner in stage.values()]
    result["iou_assertion_max_abs_delta_tied"] = max(tied or [0.0])
    result["iou_assertion_max_abs_delta_naive"] = max(naive or [0.0])
    # A monotone map cannot change a rank-defined mask; assert exactly that.
    assert result["iou_assertion_max_abs_delta_tied"] <= 1e-12, (
        f"tie-aware IoU changed under a monotone map: {result['iou_assertion_max_abs_delta_tied']}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "monotone_map.json").write_text(json.dumps(result, indent=1) + "\n")
    write_markdown(args.out / "monotone_map.md", result)
    plot_maps(args.out / "monotone_maps.png", result)
    print(f"selection check: {len(checked)} cone-stages, {n_bad} slice-count mismatches, "
          f"max |dRMSE| {rmse_d:.2e}, max |dIoU| {iou_d:.2e}")
    print(f"IoU (tie-aware, rank-defined) unchanged: max |delta| "
          f"{result['iou_assertion_max_abs_delta_tied']:.2e}")
    print(f"IoU computed naively on the mapped float32 field: max per-slice |delta| "
          f"{result['iou_assertion_max_abs_delta_naive']:.2e} (plateau tie-break)")
    print(f"wrote {args.out}/monotone_map.json, monotone_map.md, monotone_maps.png")


def table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def write_markdown(path, result):
    lines = ["# LoReLi zero-shot: monotone output map diagnostic (HAR-63)", "",
             f"Predictions: `{result['pred']}` (saved float16 cones). Truth: `x_HI = 1 - x_ion` from the raw "
             f"`.dat` cubes in `{result['data']}`.", "",
             f"Selection is `ebm21cm.evaluate_loreli`'s: border {result['border_resampled_cells']} resampled cells "
             f"-> crop index {result['crop_border_index']}, {result['crop_cells_per_side']} cells per side "
             f"({result['cells_per_slice']} per slice), LOS margin {result['los_margin']}, stage by the true slice "
             f"mean. Maps are fitted on the 28 train blocks only "
             f"({result['split_cones']['train']} cones).", "",
             "Matched floor (degraded 21cmFAST, 199 cones): native "
             + " / ".join(f"{FLOOR[s]['0.0']:.4f}" for s, _, _ in lm.STAGES)
             + "; 4.7 Mpc " + " / ".join(f"{FLOOR[s]['4.7']:.4f}" for s, _, _ in lm.STAGES) + ".",
             ""]
    for split_name in ("test", "val"):
        lines += [f"## {split_name} blocks", ""]
        rows = []
        for stage, _, _ in lm.STAGES:
            m = result["scores"][split_name][stage]
            r = m["xhi_rmse"]
            rows.append([stage, m["n_cones"], m["n_slices"],
                         f"{r['before']['0.0']:.4f}", f"{r['isotonic']['0.0']:.4f}", f"{r['logistic']['0.0']:.4f}",
                         f"{r['before']['2.3']:.4f}", f"{r['isotonic']['2.3']:.4f}", f"{r['logistic']['2.3']:.4f}",
                         f"{r['before']['4.7']:.4f}", f"{r['isotonic']['4.7']:.4f}", f"{r['logistic']['4.7']:.4f}"])
        lines += table(rows, ["stage", "n cones", "n slices", "RMSE before", "isotonic", "logistic",
                              "2.3 before", "isotonic", "logistic", "4.7 before", "isotonic", "logistic"])
        lines += ["", "Excess over the matched floor [%], native / 4.7 Mpc:", ""]
        rows = []
        for stage, _, _ in lm.STAGES:
            e = result["scores"][split_name][stage]["excess_pct_vs_floor"]
            rows.append([stage] + [f"{e[f]['0.0']:+.1f}" for f in FIELDS]
                        + [f"{e[f]['4.7']:+.1f}" for f in FIELDS])
        lines += table(rows, ["stage", "native before", "isotonic", "logistic",
                              "4.7 before", "isotonic", "logistic"])
        lines += ["", "Partially ionized cells (0.05 < x_HI < 0.95), fraction of cells:", ""]
        rows = []
        for stage, _, _ in lm.STAGES:
            p = result["scores"][split_name][stage]["partial_fraction"]
            rows.append([stage, f"{p['truth']:.3f}", f"{p['before']:.3f}", f"{p['isotonic']:.3f}",
                         f"{p['logistic']:.3f}"])
        lines += table(rows, ["stage", "truth", "before", "isotonic", "logistic"])
        lines += ["", "Corrected IoU (rank-matched at the truth's ionized fraction):", ""]
        rows = []
        for stage, _, _ in lm.STAGES:
            m = result["scores"][split_name][stage]
            rows.append([stage, f"{m['iou']['before']:.3f}", f"{m['iou_tied']['isotonic']:.3f}",
                         f"{m['iou_tied']['logistic']:.3f}"])
        lines += table(rows, ["stage", "before", "isotonic", "logistic"])
        lines += ["", "Transverse power ratio P_pred/P_true at k <= 0.6 Mpc^-1 (mean / max over the 10 bins):", ""]
        rows = []
        for stage, _, _ in lm.STAGES:
            p = result["scores"][split_name][stage]["power"]
            if "ratio_max_le_0.6" not in p:
                rows.append([stage, "n = 0", "n = 0", "n = 0"])
                continue
            rows.append([stage] + [f"{p['ratio_mean'][f]:.2f} / {p['ratio_max_le_0.6'][f]:.2f}" for f in FIELDS])
        lines += table(rows, ["stage", "before", "isotonic", "logistic"])
        lines += [""]
    lines += ["## Fitted maps", ""]
    rows = []
    for stage, _, _ in lm.STAGES:
        f = result["fit"][stage]
        if not f.get("n_cells"):
            rows.append([stage, "0", "0", "0", "-", "-"])
            continue
        rows.append([stage, f["n_cells"], f["n_slices"], f["n_cones"],
                     f"{f['isotonic']['n_knots']} knots",
                     f"a = {f['logistic']['a']:.4f}, b = {f['logistic']['b']:.4f}"])
    lines += table(rows, ["stage", "train cells", "train slices", "train cones", "isotonic", "logistic"])
    lines += ["", "a > 1 expands the predicted range, a < 1 compresses it; b shifts it. "
                  "The isotonic map is increasing and clipped to [0, 1].", ""]
    sc = result["selection_check"]
    lines += ["## Reproducibility", "",
              f"{sc['n_cone_stages']} scored cone-stages compared with their saved `cone_*.json` records: "
              f"{sc['slice_count_mismatches']} slice-count mismatches, max |dRMSE| "
              f"{sc['max_abs_rmse_diff_vs_record']:.2e}, max |dIoU| {sc['max_abs_iou_diff_vs_record']:.2e}. "
              f"{sc['note']}.", "",
              "IoU is defined on the rank of the prediction, so a monotone map cannot change it. The table above "
              "applies the map and breaks ties in the mapped field by the pre-map value; that "
              f"tie-aware IoU is identical to `before` (max per-slice |delta| "
              f"{result['iou_assertion_max_abs_delta_tied']:.2e}). Computing the mask naively on the mapped "
              "float32 field instead lets cell-index tie-breaking pick a different, equally sized set inside a "
              f"plateau; that moves IoU by at most {result['iou_assertion_max_abs_delta_naive']:.2e} per slice "
              "(per-stage worst cases in the JSON under `iou_worst_case`), an artefact rather than a change "
              "in morphology.", ""]
    path.write_text("\n".join(lines) + "\n")


def plot_maps(path, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(0.0, 1.0, 2001)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    for ax, (stage, _, _) in zip(axes, lm.STAGES):
        f = result["fit"][stage]
        ax.plot([0, 1], [0, 1], color="0.7", lw=1, ls=":", label="identity")
        if f.get("n_cells"):
            ax.step(f["isotonic"]["x"], f["isotonic"]["y"], where="post", color="C0", lw=1.5,
                    label=f"isotonic ({f['isotonic']['n_knots']} knots)")
            ax.plot(grid, logistic_apply(grid, f["logistic"]["a"], f["logistic"]["b"]), color="C3", lw=1.5,
                    label=f"logistic a={f['logistic']['a']:.2f}, b={f['logistic']['b']:.2f}")
            ax.set_title(f"{stage}\n{f['n_cells']:,} cells, {f['n_cones']} cones")
        else:
            ax.set_title(f"{stage}\nno training cells")
        ax.set_xlabel("predicted x_HI")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("mapped x_HI (fitted truth)")
    fig.suptitle("LoReLi zero-shot monotone output maps, fitted on the 28 train blocks")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
