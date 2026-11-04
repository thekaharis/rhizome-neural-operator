"""fno-21cm's 3-D evaluation suite with parallel steps and per-model result caching.

    python -m ebm21cm.suite_cached --name rhizome_vs_fno --manifests a/manifest.json b/manifest.json [--dtb]

Same outputs as the sequential ``slurm/eval3d_suite.sbatch`` it replaces, in
``runs/eval3d/<name>/``, computed differently:

* **Shared cones.** The manifests are merged on their shared cones (truth equality
  checked), exactly as before; the first model is the edge-bootstrap reference.
* **Per-model caching.** ``physical``/``physical_dtb``, ``bubble_size`` and
  ``power_spectrum`` treat every model independently, so each is run on a
  single-model manifest and cached next to that model's cubes
  (``<cubes>/suite_cache/<step>/<key>/``). The key hashes the step's arguments,
  the shared cone ids, the z grid, the model's npz paths and sizes/mtimes and the
  fno-21cm tool source, so any change recomputes. The suite then merges the
  cached results (npz keys, CSV rows, summary entries) and re-draws the figures
  with fno-21cm's own plotting functions. A comparison of one new model against
  already-evaluated ones only reads the new model's cubes.
* **Parallel steps.** Missing cache entries, both edge diagnostics (which compare
  models pairwise and are not cached) and the per-cone pages run concurrently
  (``--jobs`` processes); every step logs to ``<out>/logs/``.
* **DTB.** With ``--dtb`` the plain x_HI physical pass is skipped: the dT_b pass
  computes the same x_HI metrics (identical summary and per-cone history), so
  ``physical`` is a link to ``physical_dtb`` (whose stage table has the extra
  pre-reionization stage).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

FNO_ROOT = Path("/pfs/10/work/hd_id260-fno_training/fno-21cm")
DTB_ARGS = ["--dtb-from-npz", "--stage-edges", "0.02", "0.2", "0.4", "0.6", "0.8", "0.98", "1.001"]
CACHED = {  # step -> (module, extra args, result npz, summary json or None)
    "physical": ("viz.physical_evaluation", [], "physical_results.npz", "physical_summary.json"),
    "physical_dtb": ("viz.physical_evaluation", DTB_ARGS, "physical_results.npz", "physical_summary.json"),
    "bubble_size": ("viz.bubble_size_evaluation", [], "bubble_size_results.npz", None),
    "power_spectrum": ("viz.power_spectrum_evaluation", [], "ps_results.npz", None),
}


def merge_manifests(paths, out):
    specs = [json.loads(Path(p).read_text()) for p in paths]
    grids = [np.asarray(s["z_grid"]) for s in specs]
    if not all(np.allclose(g, grids[0]) for g in grids):
        raise SystemExit("manifests use different z grids")
    models, source = {}, {}
    for p, s in zip(paths, specs):
        for name, entries in s["models"].items():
            if name in models:
                raise SystemExit(f"duplicate model {name}")
            models[name] = {e["cone_id"]: e for e in entries}
            source[name] = Path(p).resolve().parent
    shared = sorted(set.intersection(*(set(m) for m in models.values())))
    ref = next(iter(models))
    for name, by_id in models.items():
        for c in shared:
            with np.load(models[ref][c]["npz"]) as a, np.load(by_id[c]["npz"]) as b:
                if not np.array_equal(a["truth"], b["truth"]):
                    raise SystemExit(f"truth differs for {name} cone {c}")
    merged = {"z_grid": grids[0].tolist(), "models": {n: [m[c] for c in shared] for n, m in models.items()},
              "sources": [str(p) for p in paths], "shared_cones": len(shared)}
    Path(out).write_text(json.dumps(merged, indent=1))
    return merged, source


def tool_hash(module):
    return hashlib.sha256((FNO_ROOT / (module.replace(".", "/") + ".py")).read_bytes()).hexdigest()[:16]


def cache_key(step, entries, z_grid):
    module, args, _, _ = CACHED[step]
    files = [(e["cone_id"], e["npz"], os.stat(e["npz"]).st_size, int(os.stat(e["npz"]).st_mtime)) for e in entries]
    blob = json.dumps({"step": step, "args": args, "tool": tool_hash(module), "z": z_grid, "files": files})
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def run(cmd, log, threads):
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    start = time.time()
    with open(log, "w") as f:
        code = subprocess.call(cmd, cwd=FNO_ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    return code, time.time() - start


def load_results(npz, name):
    data = np.load(npz, allow_pickle=True)
    out = {}
    for flat in data.files:
        model, _, key = flat.partition("/")
        value = data[flat]
        if key == "stage_labels":
            value = [str(v) for v in value.tolist()]
        elif value.ndim == 0:
            value = value.item()
        out[key] = value
    return out


def merge_step(step, dirs, names, out):
    """Combine single-model outputs into ``out`` and re-draw the figures."""
    module, _, npz_name, summary_name = CACHED[step]
    out.mkdir(parents=True, exist_ok=True)
    payload = {}
    for d in dirs:
        with np.load(d / npz_name, allow_pickle=True) as z:
            payload.update({k: z[k] for k in z.files})
    np.savez_compressed(out / npz_name, **payload)
    for csv_path in sorted(dirs[0].glob("*.csv")):  # concatenate rows, keeping the writer's line endings
        header = csv_path.read_text(newline="").splitlines(keepends=True)[0]
        rows = [header]
        for d in dirs:
            part = (d / csv_path.name).read_text(newline="").splitlines(keepends=True)
            if part[0] != header:
                raise SystemExit(f"{step}: CSV headers differ in {csv_path.name}")
            rows += part[1:]
        (out / csv_path.name).write_text("".join(rows), newline="")
    if summary_name:
        merged = {}
        for d in dirs:
            merged.update(json.loads((d / summary_name).read_text()))
        (out / summary_name).write_text(json.dumps(merged, indent=2))
    for cfg in dirs[0].glob("*config.json"):
        shutil.copy(cfg, out / cfg.name)
    results = {n: load_results(d / npz_name, n) for n, d in zip(names, dirs)}
    sys.path.insert(0, str(FNO_ROOT))
    if step.startswith("physical"):
        from viz import physical_evaluation as m
        m.plot_history(results, out / "physical_history.png")
        m.plot_pdfs(results, out / "physical_pdfs.png")
        if next(iter(results.values())).get("has_density"):
            m.plot_observable(results, out / "physical_21cm.png")
            m.plot_eft(results, out / "physical_eft.png")
    elif step == "bubble_size":
        from viz import bubble_size_evaluation as m
        m.plot_distributions(results, out / "bubble_size_distribution.png")
        m.plot_summary(results, out / "bubble_size_summary.png")
    else:
        from viz import power_spectrum_evaluation as m
        m.plot_overlay(results, out / "ps_overlay.png")
        m.plot_cylindrical_maps(results, out / "ps_cyl_ratio.png", out / "ps_cyl_r.png")
        m.plot_stage_curves(results, out / "ps_stage_curves.png")


def page_jobs(manifests, out):
    """(tag, native dir) for every input manifest whose model exported native cones."""
    jobs = []
    for p in manifests:
        tag = next(iter(json.loads(Path(p).read_text())["models"]))
        native = Path(p).resolve().parent / "native"
        files = sorted(native.glob("*.h5"))
        if files:
            jobs.append((tag, files))
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--dtb", action="store_true", help="model-own dT_b evaluation (multi-field exports)")
    ap.add_argument("--only-dtb", action="store_true", help="just the dT_b physical evaluation")
    ap.add_argument("--no-pages", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2, help="OMP threads per process")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    a = ap.parse_args(argv)
    dtb = a.dtb or a.only_dtb
    out = a.root / "runs" / "eval3d" / a.name
    (out / "logs").mkdir(parents=True, exist_ok=True)
    merged, source = merge_manifests(a.manifests, out / "manifest.json")
    names = list(merged["models"])
    print(f"merged {len(names)} models on {merged['shared_cones']} shared cones -> {out}", flush=True)

    steps = ["physical_dtb"] if a.only_dtb else (["physical_dtb"] if dtb else ["physical"]) + ["bubble_size",
                                                                                               "power_spectrum"]
    tasks, cache_dirs = [], {}
    for step in steps:
        module, args, npz_name, _ = CACHED[step]
        for name in names:
            entries = merged["models"][name]
            d = source[name] / "suite_cache" / step / cache_key(step, entries, merged["z_grid"])
            cache_dirs[step, name] = d
            if (d / npz_name).exists():
                print(f"[cache] {step} {name}", flush=True)
                continue
            d.mkdir(parents=True, exist_ok=True)
            single = d / "manifest.json"
            single.write_text(json.dumps({"z_grid": merged["z_grid"], "models": {name: entries}}, indent=1))
            tasks.append((f"{step}:{name}", [sys.executable, "-m", module, "--manifest", str(single), "--out", str(d)]
                          + args, out / "logs" / f"{step}__{name}.log"))
    if not a.only_dtb:
        for mode in ("slice", "3d"):
            tasks.append((f"edge_{mode}", [sys.executable, "-m", "viz.boundary_band_diagnostic", "--manifest",
                                           str(out / "manifest.json"), "--mode", mode, "--reference", names[0],
                                           "--out", str(out / f"edge_{mode}")], out / "logs" / f"edge_{mode}.log"))
        if not a.no_pages:
            for tag, files in page_jobs(a.manifests, out):
                for tool, sub, extra in (("viz.multifield_detailed", "detailed", ["--note", tag]),
                                         ("viz.multifield_lightcone_strips", "strips", ["--crop", "--note", tag]),
                                         ("viz.multifield_slices", "slices", ["--n-slices", "5"])):
                    tasks.append((f"{sub}:{tag}", [sys.executable, "-m", tool, "--prediction", *map(str, files),
                                                   "--out-dir", str(out / sub / tag)] + extra,
                                  out / "logs" / f"{sub}__{tag}.log"))
    print(f"{len(tasks)} tasks, {a.jobs} at a time", flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = {pool.submit(run, cmd, log, a.threads): label for label, cmd, log in tasks}
        for fut, label in futures.items():
            code, seconds = fut.result()
            print(f"{'ok ' if code == 0 else 'FAILED'} {label} ({seconds / 60:.1f} min)", flush=True)
            if code:
                failed.append(label)
    for step in steps:
        dirs = [cache_dirs[step, n] for n in names]
        if all((d / CACHED[step][2]).exists() for d in dirs):
            merge_step(step, dirs, names, out / step)
            print(f"merged {step}", flush=True)
    if dtb and not a.only_dtb:
        link = out / "physical"
        if link.is_symlink() or link.exists():
            shutil.rmtree(link) if link.is_dir() and not link.is_symlink() else link.unlink()
        link.symlink_to("physical_dtb")
    print(("FAILED: " + ", ".join(failed)) if failed else f"eval3d suite done: {out}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
