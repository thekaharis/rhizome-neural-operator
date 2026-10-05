"""Full-cone inference timing for the multi-field models (x_HI + T_b), stage by stage.

    python -m ebm21cm.bench_inference_mf --cone 1751 --out runs/eval3d/bench_inference_mf_a100.json \\
        --rhizome2d runs/rhizome_mf2d_w64_structured_all \\
        --rhizome3d runs/rhizome3d_mf/C_histtb_structured_cont10 --fno <fno multi-field best.pt>

One test cone of the cleaned multi-field preparation, every native slice, as in
the exports. Stages, each timed with CUDA synchronisation after a warm-up:

* 2-D slice-wise rhizome: (a) fno-21cm window reads, (b) input building on the
  CPU (LOS bands, history band means, T_b physical factor; includes (a)),
  (c) GPU forward + T_b assembly incl. host->device copies, for several batch
  sizes in float32 and bfloat16 autocast (bf16 checked against fp32).
* 3-D rhizome / fno-21cm: (a) window reads (their whole input prep), (c) forward.

Medians of ``--repeats`` runs; peak GPU memory of the forward stage.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .data.mf_slices import FNO_ROOT, PREPARATION, fno_dataset
from .export_slicewise_mf import iter_slice_inputs_gpu, slice_inputs, slice_inputs_gpu
from .model.rhizome import RhizomeOperator2d
from .train_rhizome_mf2d import tb_normalized


def timed(fn, device, repeats):
    fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize(device)
        t.append(time.perf_counter() - start)
    return float(np.median(t)), torch.cuda.max_memory_allocated(device) / 2**30, out


def wall(fn, repeats):
    t = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        t.append(time.perf_counter() - start)
    return float(np.median(t)), out


def bench_2d(run, sid, device, batches, repeats):
    meta = json.loads((run / "metadata.json").read_text())
    attrs = meta["cache_attrs"]
    bands = [tuple(b) for b in json.loads(attrs["bands"])]
    wide = [tuple(b) for b in json.loads(attrs.get("wide_bands", "[]"))]
    dataset, rows, LOSWindowConfig, head, _, param_names = fno_dataset(PREPARATION, FNO_ROOT, attrs["history"])
    row = next(r for r in rows["test"] if int(re.findall(r"\d+", Path(dataset.file_paths[r]).stem)[-1]) == sid)
    n = len(dataset.redshifts[row])
    ip = [list(dataset.channel_names).index(p) for p in param_names]
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model = RhizomeOperator2d(**best["model_config"]).to(device).eval()
    model.load_state_dict(best["model"])
    tb_off, tb_scale = best["tb_norm"]

    # (a) window reads alone: the same chunked windows slice_inputs requests.
    from .data.cache import band_extent
    below, above = band_extent(bands + wide)
    halo = max(below, above + 1)
    cfg = LOSWindowConfig(mode="contiguous", size=512 + 2 * halo, halo=halo, windows_per_cone=1)
    read_s, _ = wall(lambda: [dataset.window(row, s - halo, cfg) for s in range(0, n, 512)], repeats)
    # (b) full CPU input building (includes the reads).
    prep_s, inp = wall(lambda: slice_inputs(dataset, row, np.arange(n), head, LOSWindowConfig, bands, ip, wide=wide),
                       repeats)
    cond = torch.from_numpy(inp["cond"]).pin_memory()
    scal = torch.from_numpy(inp["scalars"]).pin_memory()
    phys = torch.from_numpy(inp["phys"][:, None]).pin_memory()

    def forward(batch, bf16):
        xs, ts = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            for a in range(0, n, batch):
                b = slice(a, a + batch)
                out = model(cond[b].to(device, non_blocking=True), scal[b].to(device, non_blocking=True)).float()
                t = tb_normalized(out, phys[b].to(device, non_blocking=True), best["head"], tb_off, tb_scale)
                xs.append(out[:, 0].sigmoid())
                ts.append(t[:, 0] * tb_scale + tb_off)
        return torch.cat(xs), torch.cat(ts)

    result = {"slices": n, "params": sum(p.numel() for p in model.parameters()),
              "window_read_s": read_s, "input_prep_cpu_s": prep_s, "forward": {}}

    # (b') the same inputs built on the GPU (includes the reads), checked against the CPU path.
    def gpu_prep():
        out = slice_inputs_gpu(dataset, row, np.arange(n), head, LOSWindowConfig, bands, ip, device, wide=wide)
        torch.cuda.synchronize(device)
        return out
    gprep_s, ginp = wall(gpu_prep, repeats)
    result["input_prep_gpu_s"] = gprep_s
    result["gpu_vs_cpu_inputs_max_abs"] = {k: float((ginp[k].cpu() - torch.from_numpy(inp[k])).abs().max())
                                           for k in ("cond", "scalars", "phys")}
    print("gpu prep", gprep_s, result["gpu_vs_cpu_inputs_max_abs"], flush=True)

    def end_to_end(batch=128):
        g = slice_inputs_gpu(dataset, row, np.arange(n), head, LOSWindowConfig, bands, ip, device, wide=wide)
        with torch.no_grad():
            for a in range(0, n, batch):
                b = slice(a, a + batch)
                out = model(g["cond"][b], g["scalars"][b])
                tb_normalized(out, g["phys"][b][:, None], best["head"], tb_off, tb_scale)
        torch.cuda.synchronize(device)
    result["end_to_end_gpu_inputs_s"], _ = wall(end_to_end, repeats)
    print("end to end (GPU inputs, fp32 b128)", result["end_to_end_gpu_inputs_s"], flush=True)

    def pipelined(batch=128):
        with torch.no_grad():
            for g in iter_slice_inputs_gpu(dataset, row, np.arange(n), head, LOSWindowConfig, bands, ip, device,
                                           wide=wide):
                for a in range(0, len(g["cond"]), batch):
                    b = slice(a, a + batch)
                    out = model(g["cond"][b], g["scalars"][b])
                    tb_normalized(out, g["phys"][b][:, None], best["head"], tb_off, tb_scale)
        torch.cuda.synchronize(device)
    result["end_to_end_pipelined_s"], _ = wall(pipelined, repeats)
    print("end to end (GPU inputs, prefetch + per-chunk forward)", result["end_to_end_pipelined_s"], flush=True)
    del ginp
    reference = None
    for bf16 in (False, True):
        for batch in batches:
            key = f"{'bf16' if bf16 else 'fp32'}_b{batch}"
            try:
                s, mem, (x, t) = timed(lambda: forward(batch, bf16), device, repeats)
            except (RuntimeError, ValueError) as e:  # e.g. kernels without bf16 support
                result["forward"][key] = {"error": str(e).splitlines()[0][:200]}
                torch.cuda.empty_cache()
                continue
            entry = {"seconds": s, "slices_per_s": n / s, "peak_gib": mem}
            if reference is None:
                reference = (x.float(), t.float())
            else:
                entry["max_abs_dx_vs_fp32"] = float((x.float() - reference[0]).abs().max())
                entry["rmse_dtb_vs_fp32_mk"] = float((t.float() - reference[1]).square().mean().sqrt())
            result["forward"][key] = entry
            print(key, entry, flush=True)
    del cond, scal, phys
    return result


def bench_3d(kind, source, sid, device, repeats):
    from .export_cubes3d import load_model

    args = SimpleNamespace(fno_root=FNO_ROOT, fno=source if kind == "fno" else None,
                           rhizome=source if kind == "rhizome" else None)
    _, model, dataset, rows, window, _ = load_model(args, device)
    row = next(r for r in rows["test"] if int(re.findall(r"\d+", Path(dataset.file_paths[r]).stem)[-1]) == sid)
    n = len(dataset.redshifts[row])
    read_s, samples = wall(lambda: [dataset.window(row, s - window.halo, window) for s in range(0, n, window.core)],
                           repeats)
    xs = [s["x"][None].pin_memory() for s in samples]
    ctx = [s["context"][None].pin_memory() if "context" in s else None for s in samples]

    def run_all():
        with torch.no_grad():
            for x, c in zip(xs, ctx):
                kwargs = {"context": c.to(device, non_blocking=True)} if c is not None else {}
                model(x.to(device, non_blocking=True), **kwargs)

    s, mem, _ = timed(run_all, device, repeats)
    return {"slices": n, "windows": len(xs), "params": sum(p.numel() for p in model.parameters()),
            "window_read_s": read_s, "forward": {"default": {"seconds": s, "slices_per_s": n / s, "peak_gib": mem}}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cone", type=int, default=1751)
    ap.add_argument("--rhizome2d", type=Path, required=True)
    ap.add_argument("--rhizome3d", required=True)
    ap.add_argument("--fno", required=True)
    ap.add_argument("--batch", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-3d", action="store_true", help="only the 2-D model (keeps earlier 3-D results in --out)")
    a = ap.parse_args(argv)
    device = torch.device("cuda")
    results = json.loads(a.out.read_text()) if a.skip_3d and a.out.exists() else {}
    results.update({"gpu": torch.cuda.get_device_name(device), "cone": a.cone})
    results["rhizome2d_slicewise_mf"] = bench_2d(a.rhizome2d, a.cone, device, a.batch, a.repeats)
    a.out.write_text(json.dumps(results, indent=1))
    if a.skip_3d:
        return
    results["rhizome3d_mf_C"] = bench_3d("rhizome", a.rhizome3d, a.cone, device, a.repeats)
    print("3-D rhizome", results["rhizome3d_mf_C"], flush=True)
    a.out.write_text(json.dumps(results, indent=1))
    results["fno_mf_tbs_histtb"] = bench_3d("fno", a.fno, a.cone, device, a.repeats)
    print("fno", results["fno_mf_tbs_histtb"], flush=True)
    a.out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
