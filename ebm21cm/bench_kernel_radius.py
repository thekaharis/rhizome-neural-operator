"""Cost of the compact real-space kernel against the Fourier kernel, on the multi-field 2-D model's shapes.

    python -m ebm21cm.bench_kernel_radius --out runs/eval3d/kernel_radius_bench_a100.json

The 2-D multi-field rhizome (26 band channels, 38 scalars, width 64, 6 tied
updates) on 140 x 140 slices, with the kernel either ``SpectralConv2d`` (modes
24) or ``LocalConv2d`` of radius R. For each kernel, timed with CUDA
synchronisation after a warm-up (medians of ``--repeats``), PyTorch's default
precision settings (cuDNN convolutions in TF32, FFT/einsum in float32), as in
training:

* the kernel alone, forward, at the training and inference batch sizes;
* a full training step (forward, backward, Adam) at the training batch size;
* model inference (forward) at the inference batch size.

Also the analytic multiply-accumulate count of one kernel application per
slice: Fourier ~ C*N*log2(N) (forward + inverse real FFT, 2.5 N log2 N real
operations per channel each, counted as MACs/2) plus 4 C^2 per retained mode;
compact C^2 * n_offsets * N (dense) or (2 C r + r n) * N (rank r).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .model.rhizome import RhizomeOperator2d, disc_offsets

IN_CHANNELS, SCALARS, OUT = 26, 38, 2


def timed(fn, repeats, device):
    fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        t.append(time.perf_counter() - start)
    return float(np.median(t)), torch.cuda.max_memory_allocated(device) / 2**30


def kernel_macs(width, size, modes=None, radius=None, rank=None):
    n = size * size
    if radius is None:
        fft = 2 * 2.5 * n * math.log2(n) * width / 2
        retained = modes + (modes - 1) + modes * (modes - 1) + (modes - 1) ** 2  # dc, axis, positive, negative
        return fft + 4 * width ** 2 * retained
    k = len(disc_offsets(radius))
    return n * (width ** 2 * k if rank is None else 2 * width * rank + rank * k)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--radii", type=float, nargs="+", default=[1, 2 ** 0.5, 2, 5 ** 0.5, 8 ** 0.5, 3, 4, 5, 6, 8])
    ap.add_argument("--ranks", type=int, nargs="*", default=[16])
    ap.add_argument("--rank-radii", type=float, nargs="*", default=[2, 4, 8])
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--modes", type=int, default=24)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--size", type=int, default=140)
    ap.add_argument("--train-batch", type=int, default=32)
    ap.add_argument("--infer-batch", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--cudnn-benchmark", action="store_true", help="let cuDNN time its algorithms per shape")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    torch.backends.cudnn.benchmark = a.cudnn_benchmark
    device = torch.device("cuda")
    variants = [("spectral_m%d" % a.modes, {})]
    variants += [(f"local_R{r:.3g}", {"radius": r}) for r in a.radii]
    variants += [(f"local_R{r:.3g}_rank{k}", {"radius": r, "rank": k}) for k in a.ranks for r in a.rank_radii]
    results = {"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "size": a.size,
               "width": a.width, "steps": a.steps, "train_batch": a.train_batch, "infer_batch": a.infer_batch,
               "cudnn_tf32": torch.backends.cudnn.allow_tf32, "cudnn_benchmark": a.cudnn_benchmark, "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
               "variants": {}}
    for name, extra in variants:
        torch.manual_seed(0)
        model = RhizomeOperator2d(IN_CHANNELS, OUT, SCALARS, a.width, a.modes, a.steps, 0.5, **extra).to(device)
        kernel = model.cells[0].interaction.kernel
        entry = {"params_model": sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()),
                 "params_kernel": sum(p.numel() * (2 if p.is_complex() else 1) for p in kernel.parameters()),
                 "offsets": len(disc_offsets(extra["radius"])) if "radius" in extra else None,
                 "reach_cells": a.steps * extra["radius"] if "radius" in extra else None,
                 "kernel_gmacs_per_slice": kernel_macs(a.width, a.size, a.modes, extra.get("radius"),
                                                       extra.get("rank")) / 1e9}
        try:
            with torch.no_grad():
                for batch in (a.train_batch, a.infer_batch):
                    h = torch.randn(batch, a.width, a.size, a.size, device=device)
                    entry[f"kernel_fwd_b{batch}_ms"] = 1e3 * timed(lambda: kernel(h), a.repeats, device)[0]
                    del h
            cond = torch.randn(a.train_batch, IN_CHANNELS, a.size, a.size, device=device)
            scal = torch.randn(a.train_batch, SCALARS, device=device)
            target = torch.rand(a.train_batch, OUT, a.size, a.size, device=device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)

            def train_step():
                opt.zero_grad(set_to_none=True)
                torch.nn.functional.mse_loss(model(cond, scal), target).backward()
                opt.step()
            model.train()
            entry["train_step_ms"], entry["train_peak_gib"] = timed(train_step, a.repeats, device)
            entry["train_step_ms"] *= 1e3
            model.eval()
            cond = torch.randn(a.infer_batch, IN_CHANNELS, a.size, a.size, device=device)
            scal = torch.randn(a.infer_batch, SCALARS, device=device)
            with torch.no_grad():
                s, mem = timed(lambda: model(cond, scal), a.repeats, device)
            entry["infer_ms"], entry["infer_peak_gib"] = 1e3 * s, mem
            entry["infer_slices_per_s"] = a.infer_batch / s
        except torch.cuda.OutOfMemoryError as e:
            entry["error"] = str(e).splitlines()[0][:200]
        results["variants"][name] = entry
        print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in entry.items()}), flush=True)
        del model, kernel
        torch.cuda.empty_cache()
        a.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
