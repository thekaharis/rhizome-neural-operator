"""Resolution-only part of the LoReLi transfer: 21cmFAST test cones degraded to LoReLi's grid.

    python -m ebm21cm.evaluate_degraded --run runs/rhizome_mf2d_w64_structured_all \
        --cache /pfs/.../rhizome_mf_slices_all.h5 --mirror /pfs/.../native_mirror_4f \
        --out runs/eval3d/loreli/degraded

For each selected 21cmFAST test cone the lightcone is re-expressed on LoReLi's
128 cells / 2.297 Mpc and 1536 slices / 1.64 cMpc (density, x_HI, T_b and the
stored velocity all degraded the same way), pushed through the same LoReLi
adapter and remapped model, and compared with the native 21cmFAST prediction
on the same slices. The gap between the two is the resolution-only part of the
transfer error; the remaining gap on real LoReLi is physics and parameters.

Metrics are x_HI RMSE at native resolution, after 2.3 and 4.7 Mpc smoothing,
IoU at matched ionized fraction, and T_b RMSE, per stage, on the central
200 Mpc (87 native cells) where the 21cmFAST box sits inside the 294 Mpc plane.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from . import loreli_metrics as lm
from .data.degraded import Degraded21cmCone
from .data.loreli import TRAINING_CELL_MPC, area_matrix, pad_to, training_stats
from .loreli_inference import downsample_transverse, load_model, predict
from .train_rhizome_mf2d import tb_normalized

SIGMAS = (0.0, 2.3, 4.7)


def pad_to_loreli(plane, n_out, n_native):
    """Map a 140-cell 21cmFAST plane into the central 200 Mpc of the n_native LoReLi grid."""
    weights = area_matrix(n_native, n_out)
    return (weights @ pad_to(plane, n_out) @ weights.T).astype(np.float32)


class Acc:
    def __init__(self):
        self.n = self.cells = 0
        self.se = {s: 0.0 for s in SIGMAS}
        self.tb_se = 0.0
        self.iou = 0.0
        self.iou_n = 0

    def add(self, pred, true, tb_pred, tb_true, cell):
        self.n += 1
        self.cells += true.size
        for s in SIGMAS:
            self.se[s] += float(np.sum((lm.smooth(pred, s, cell) - lm.smooth(true, s, cell)) ** 2))
        self.tb_se += float(np.sum((tb_pred - tb_true) ** 2))
        iou = lm.iou_matched(pred, true)
        if np.isfinite(iou):
            self.iou += iou
            self.iou_n += 1

    def metrics(self):
        return {"n_slices": self.n,
                "xhi_rmse": {str(s): float(np.sqrt(self.se[s] / max(self.cells, 1))) for s in SIGMAS},
                "iou_matched": float(self.iou / self.iou_n) if self.iou_n else float("nan"),
                "tb_rmse_mk": float(np.sqrt(self.tb_se / max(self.cells, 1)))}


def select_rows(h, cone_ids, per_stage):
    """Cache rows of the selected cones, evenly sampled per x_HI stage."""
    rows = np.flatnonzero(h["split"][:] == 2)
    cone = h["cone_id"][:][rows]
    z = h["z"][:][rows]
    xhi = np.array([h["xhi"][r].astype(np.float32).mean() for r in rows])
    picked = []
    for cid in cone_ids:
        for name, lo, hi in lm.STAGES:
            mine = rows[(cone == cid) & (xhi >= lo) & (xhi < hi) & (z > 5.05) & (z < 14.95)]
            mine = mine[np.argsort(z[np.isin(rows, mine)])]
            if len(mine):
                pick = np.unique(np.linspace(0, len(mine) - 1, min(per_stage, len(mine))).round().astype(int))
                picked += [(int(cid), int(r), name) for r in mine[pick]]
    return picked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--mirror", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cones", type=int, default=20)
    ap.add_argument("--per-stage", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    stats = training_stats(args.cache)
    native, model, best = load_model(args.run)
    device = torch.device(args.device)
    model = model.to(device)
    tb_offset, tb_scale = best["tb_norm"]

    with h5py.File(args.cache, "r") as h:
        test_cones = np.unique(h["cone_id"][:][h["split"][:] == 2])[:args.cones]
        picked = select_rows(h, test_cones, args.per_stage)
        rows = []
        for cid, r, stage in picked:
            rows.append({"cid": cid, "row": r, "stage": stage, "z": float(h["z"][r]),
                         "cond": np.asarray(h["cond"][r], np.float32), "scalars": np.asarray(h["scalars"][r], np.float32),
                         "phys": np.asarray(h["phys"][r], np.float32), "xhi": np.asarray(h["xhi"][r], np.float32),
                         "tb": np.asarray(h["tb"][r], np.float32) * tb_scale + tb_offset})
    print(f"{len(rows)} slices from {len(test_cones)} test cones", flush=True)

    accum = {name: {"native": Acc(), "degraded": Acc()} for name, _, _ in lm.STAGES}
    per_cone = []
    for k, cid in enumerate(test_cones):
        mirror = args.mirror / f"21cmfast_11d_sample{int(cid):06d}.h5"
        cone = Degraded21cmCone(mirror)
        from .data.loreli import LoreliAdapter

        adapter = LoreliAdapter(cone, stats, velocity="stored")
        mine = [r for r in rows if r["cid"] == cid]
        record = {"sim": int(cid), "n": len(mine), "stages": {}}
        for item in mine:
            i = int(np.argmin(np.abs(adapter.z_target - item["z"])))
            with torch.no_grad():
                out = native(torch.as_tensor(item["cond"])[None], torch.as_tensor(item["scalars"])[None])
                native_x = out[0, 0].sigmoid().numpy()
                native_tb = (tb_normalized(out, torch.as_tensor(item["phys"])[None, None], best["head"],
                                           tb_offset, tb_scale)[0, 0] * tb_scale + tb_offset).numpy()
                inp = adapter.inputs(i, i + 1)
                # Resolution-only: keep the row's parameter and history scalars, degrade only the fields.
                inp["scalars"][0] = item["scalars"]
                xd, td = predict(model, inp, best, device)
            central = _central(cone.n_native)
            truth_x = pad_to_loreli(item["xhi"], int(round(cone.l_perp / TRAINING_CELL_MPC)), cone.n_native)[central, central]
            truth_tb = pad_to_loreli(item["tb"], int(round(cone.l_perp / TRAINING_CELL_MPC)), cone.n_native)[central, central]
            native_x = pad_to_loreli(native_x, int(round(cone.l_perp / TRAINING_CELL_MPC)), cone.n_native)[central, central]
            native_tb = pad_to_loreli(native_tb, int(round(cone.l_perp / TRAINING_CELL_MPC)), cone.n_native)[central, central]
            deg_x = downsample_transverse(xd[0], adapter, cone)[central, central]
            deg_tb = downsample_transverse(td[0], adapter, cone)[central, central]
            cell = cone.l_perp / cone.n_native
            accum[item["stage"]]["native"].add(native_x, truth_x, native_tb, truth_tb, cell)
            accum[item["stage"]]["degraded"].add(deg_x, truth_x, deg_tb, truth_tb, cell)
            record["stages"].setdefault(item["stage"], {"native": Acc(), "degraded": Acc()})
            record["stages"][item["stage"]]["native"].add(native_x, truth_x, native_tb, truth_tb, cell)
            record["stages"][item["stage"]]["degraded"].add(deg_x, truth_x, deg_tb, truth_tb, cell)
        record["stages"] = {name: {k: a.metrics() for k, a in v.items()} for name, v in record["stages"].items()}
        per_cone.append(record)
        print(f"[{k + 1}/{len(test_cones)}] cone {int(cid)}: {len(mine)} slices", flush=True)

    summary = {"run": str(args.run.resolve()), "n_cones": len(test_cones),
               "stages": {name: {k: a.metrics() for k, a in v.items()} for name, v in accum.items()}}
    (args.out / "degraded.json").write_text(json.dumps({"summary": summary, "cones": per_cone}, indent=1))
    lines = [f"# 21cmFAST degraded to LoReLi resolution: {args.run.name}", "",
             "| stage | n | native RMSE | degraded RMSE | native 4.7 | degraded 4.7 | native IoU | degraded IoU |",
             "|---|---|---|---|---|---|---|---|"]
    for name, _, _ in lm.STAGES:
        n, d = summary["stages"][name]["native"], summary["stages"][name]["degraded"]
        lines.append(f"| {name} | {n['n_slices']} | {n['xhi_rmse']['0.0']:.4f} | {d['xhi_rmse']['0.0']:.4f} | "
                     f"{n['xhi_rmse']['4.7']:.4f} | {d['xhi_rmse']['4.7']:.4f} | {n['iou_matched']:.3f} | "
                     f"{d['iou_matched']:.3f} |")
    (args.out / "degraded.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def _central(n):
    """Central 200 Mpc of the 294 Mpc LoReLi plane, in native 128 cells (87 cells)."""
    size = int(round(200.0 / (294.0 / n)))
    start = (n - size) // 2
    return slice(start, start + size)


if __name__ == "__main__":
    main()
