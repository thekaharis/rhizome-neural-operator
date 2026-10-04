"""Full-cone x_HI inference timing: 2-D slice-wise rhizome vs 3-D LOS-window models.

    python -m ebm21cm.bench_inference --cone 1063 --rhizome2d runs/rhizome_xhi_w128 \\
        --rhizome3d runs/rhizome3d/w48_m36_h200_cont10 --fno <fno best.pt>

For one test cone at native LOS resolution, each model predicts every slice as
in its export (``export_slicewise`` / ``predict_native_cone``). Timings exclude
lightcone I/O: inputs are prepared on the CPU first (2-D: the 13 density bands
of every slice; 3-D: every LOS window), then only host->device transfer and the
forward passes are timed, after a warm-up, with CUDA synchronisation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from .data.cache import Normalizer, read_stats
from .data.lightcone import Lightcone
from .export_slicewise import band_stack
from .model.rhizome import RhizomeOperator2d


def timed(fn, device, repeats):
    fn()  # warm-up (cuDNN/cuFFT plans, allocator)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        t.append(time.perf_counter() - start)
    return float(np.median(t)), torch.cuda.max_memory_allocated(device) / 2**30


def bench_2d(run, raw, device, batch, repeats):
    meta = json.loads((run / "metadata.json").read_text())
    stats = read_stats(meta["cache"])
    with h5py.File(meta["cache"], "r") as h:
        bands = [tuple(b) for b in json.loads(h.attrs["bands"])]
        names = json.loads(h.attrs["param_names"])
    model = RhizomeOperator2d(**meta["model_config"]).to(device).eval()
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=False)["model"])
    with Lightcone(raw) as lc:
        density = lc.read_range("density", 0, lc.n_los)
        z, params = np.asarray(lc.redshifts), lc.params(names, missing="nan")
    start = time.perf_counter()
    cond, _ = band_stack(density, np.arange(len(z)), bands)
    band_seconds = time.perf_counter() - start
    mean, std = float(stats["delta_log1p_mean"]), float(stats["delta_log1p_std"])
    cond = torch.from_numpy(cond).pin_memory()
    scal = torch.from_numpy(Normalizer(stats).scalars(z, np.tile(params, (len(z), 1))).astype(np.float32))

    def run_all():
        with torch.no_grad():
            for a in range(0, len(z), batch):
                c = cond[a:a + batch].to(device, non_blocking=True)
                c = (torch.log1p(c.clamp_min(-0.99)) - mean) / std
                model(c, scal[a:a + batch].to(device)).sigmoid()

    seconds, mem = timed(run_all, device, repeats)
    return {"slices": len(z), "seconds": seconds, "band_prep_cpu_seconds": band_seconds, "peak_gib": mem,
            "params": sum(p.numel() for p in model.parameters()), "batch": batch}


def bench_3d(kind, source, sample_id, device, repeats, fno_root):
    from .export_cubes3d import load_model

    args = SimpleNamespace(fno_root=fno_root, fno=source if kind == "fno" else None,
                           rhizome=source if kind == "rhizome" else None)
    _, model, dataset, rows, window, _ = load_model(args, device)
    row = next(r for r in rows["test"] if int(Path(dataset.file_paths[r]).stem[-6:]) == sample_id)
    n = len(dataset.redshifts[row])
    samples = [dataset.window(row, start - window.halo, window) for start in range(0, n, window.core)]
    xs = [s["x"][None].pin_memory() for s in samples]
    ctx = [s["context"][None].pin_memory() if "context" in s else None for s in samples]

    def run_all():
        with torch.no_grad():
            for x, c in zip(xs, ctx):
                kwargs = {"context": c.to(device, non_blocking=True)} if c is not None else {}
                model(x.to(device, non_blocking=True), **kwargs)

    seconds, mem = timed(run_all, device, repeats)
    return {"slices": n, "windows": len(xs), "window": [window.size, window.halo], "seconds": seconds,
            "peak_gib": mem, "params": sum(p.numel() for p in model.parameters())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cone", type=int, default=1063)
    ap.add_argument("--rhizome2d", type=Path, required=True)
    ap.add_argument("--rhizome3d", required=True)
    ap.add_argument("--fno", required=True)
    ap.add_argument("--raw", default="/pfs/10/work/hd_id260-fno_training/data/data/21cmfast_11d_sample{:06d}.h5")
    ap.add_argument("--fno-root", default="/pfs/10/work/hd_id260-fno_training/fno-21cm")
    ap.add_argument("--batch", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(device)
    results = {"gpu": gpu, "cone": args.cone}
    for b in args.batch:
        results[f"rhizome2d_slicewise_b{b}"] = bench_2d(args.rhizome2d, args.raw.format(args.cone), device, b,
                                                        args.repeats)
        print(b, results[f"rhizome2d_slicewise_b{b}"], flush=True)
    results["rhizome3d"] = bench_3d("rhizome", args.rhizome3d, args.cone, device, args.repeats, args.fno_root)
    print(results["rhizome3d"], flush=True)
    results["fno3d"] = bench_3d("fno", args.fno, args.cone, device, args.repeats, args.fno_root)
    print(results["fno3d"], flush=True)
    for v in results.values():
        if isinstance(v, dict):
            v["slices_per_s"] = v["slices"] / v["seconds"]
    args.out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
