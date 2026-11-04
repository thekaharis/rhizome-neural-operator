"""Time the non-model stages of one LoReLi cone: read, adapter, inputs, map and metrics.

    python scripts/profile_loreli.py --data /pfs/.../loreli_lc128 \
        --cache /pfs/.../rhizome_mf_slices_all.h5 --sim 19799 --device cpu

The model is not run here; the GPU model cost is measured separately (or
estimated from the 140-cell benchmark). This isolates the CPU/IO work that
dominated the first 383 s/cone evaluation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ebm21cm import loreli_metrics as lm  # noqa: E402
from ebm21cm.data.loreli import LoreliAdapter, LoreliCone, read_cube, training_stats  # noqa: E402
from ebm21cm.evaluate_loreli import Accumulator, select  # noqa: E402
from ebm21cm.loreli_inference import map_to_native  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--sim", type=int, default=19799)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shape-per-stage", type=int, default=48)
    ap.add_argument("--baseline-per-stage", type=int, default=32)
    args = ap.parse_args(argv)
    path = args.data / f"lightcone_xa_tpHI_xion_dens_dtb_no_rsd_simu{args.sim}.dat"
    stats = training_stats(args.cache)
    times = {}

    t = time.perf_counter()
    raw = read_cube(path)
    times["read (contiguous)"] = time.perf_counter() - t
    t = time.perf_counter()
    cone = LoreliCone(path, data=raw)
    times["cone init + validity + history"] = time.perf_counter() - t
    t = time.perf_counter()
    adapter = LoreliAdapter(cone, stats, device=args.device)
    times["adapter init + velocity FFT"] = time.perf_counter() - t
    t = time.perf_counter()
    for a in range(0, len(adapter.target_distance), args.batch):
        adapter.inputs(a, min(a + args.batch, len(adapter.target_distance)))
    times["all model inputs"] = time.perf_counter() - t
    fake = np.random.default_rng(0).random((len(adapter.target_distance), 280, 280)).astype(np.float32)
    t = time.perf_counter()
    pred_x, pred_tb, keep = map_to_native(fake, fake, adapter, cone, 4)
    times["map to native"] = time.perf_counter() - t

    truth_raw = cone.read(0, cone.n_los)
    truth = 1.0 - truth_raw["xion"]
    tb = truth_raw["dtb"]
    delta = cone.delta(truth_raw["dens"])
    cell = cone.l_perp / cone.n_native
    border = 4
    crop = slice(border, cone.n_native - border)
    stages = {name: Accumulator() for name, _, _ in lm.STAGES}
    active = {name: [] for name in stages}
    t = time.perf_counter()
    for i in np.flatnonzero(cone.valid):
        true = truth[crop, crop, i]
        stage = lm.stage_of(float(true.mean()))
        if stage is None:
            continue
        active[stage].append(int(i))
        stages[stage].add_slice(pred_x[i, crop, crop], true, pred_tb[i, crop, crop], tb[crop, crop, i], cell)
    times["main metrics"] = time.perf_counter() - t
    t = time.perf_counter()
    for name, idx in active.items():
        for i in select(idx, args.shape_per_stage):
            stages[name].add_shape(pred_x[i, crop, crop], truth[crop, crop, i], pred_tb[i, crop, crop],
                                   tb[crop, crop, i], cell)
    times["shape metrics"] = time.perf_counter() - t
    t = time.perf_counter()
    for name, idx in active.items():
        for i in select(idx, args.baseline_per_stage):
            true = truth[crop, crop, i]
            score = delta[crop, crop, i]
            fraction = float((true < 0.5).mean())
            if not 0 < fraction < 1:
                continue
            threshold = np.quantile(score, 1 - fraction)
            stages[name].add_baseline("density", np.where(score >= threshold, 0.0, 1.0), true)
            es = lm.sharp_k_max(score, cell)
            threshold = np.quantile(es, 1 - fraction)
            stages[name].add_baseline("excursion", np.where(es >= threshold, 0.0, 1.0), true)
    times["baselines"] = time.perf_counter() - t

    print(f"cone {args.sim}: {len(adapter.target_distance)} target slices, "
          f"active { {k: len(v) for k, v in active.items()} }")
    for name, value in times.items():
        print(f"  {name:34s} {value:8.2f} s")
    print(f"  {'non-model total':34s} {sum(times.values()):8.2f} s")
    print("Add the GPU model time separately; at stride N the inputs/model/metrics scale by about 1/N.")


if __name__ == "__main__":
    main()
