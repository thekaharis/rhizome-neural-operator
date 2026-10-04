"""Re-render an eval3d suite's figures for a subset of its models, with new labels,
from the saved result files (no recomputation from the cubes).

    python -m ebm21cm.replot_suite --suite runs/eval3d/slicewise2d_mf_vs_3d_mf \\
        --out runs/eval3d/h2h_rhizome_vs_ufno \\
        --model rhizome2d_mf_w64=Rhizome --model mf_cnn_whno_es_tbs_histtb_tbclean_ep26=UFNO

Uses fno-21cm's own plotting functions on ``physical*/physical_results.npz``,
``power_spectrum/ps_results.npz`` and ``bubble_size/bubble_size_results.npz``.
The edge diagnostics save only their tables, and the per-cone pages are
per-model renders, so those are not covered here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

FNO_ROOT = "/pfs/10/work/hd_id260-fno_training/fno-21cm"


def load(npz, rename):
    data = np.load(npz, allow_pickle=True)
    results = {}
    for flat in data.files:
        name, _, key = flat.partition("/")
        if name not in rename:
            continue
        value = data[flat]
        if key == "stage_labels":
            value = [str(v) for v in value.tolist()]
        elif value.ndim == 0:
            value = value.item()
        results.setdefault(rename[name], {})[key] = value
    missing = set(rename.values()) - set(results)
    if missing:
        raise SystemExit(f"{npz}: no results for {sorted(missing)}")
    return {label: results[label] for label in rename.values()}  # keep the requested order


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, help="suite_name=label (order kept)")
    ap.add_argument("--fno-root", default=FNO_ROOT)
    a = ap.parse_args(argv)
    rename = dict(m.split("=", 1) for m in a.model)
    sys.path.insert(0, a.fno_root)
    from viz import bubble_size_evaluation as bubble
    from viz import physical_evaluation as phys
    from viz import power_spectrum_evaluation as ps

    for sub in ("physical", "physical_dtb"):
        npz = a.suite / sub / "physical_results.npz"
        if not npz.exists():
            continue
        results = load(npz, rename)
        out = a.out / sub
        out.mkdir(parents=True, exist_ok=True)
        phys.plot_history(results, out / "physical_history.png")
        phys.plot_pdfs(results, out / "physical_pdfs.png")
        if next(iter(results.values())).get("has_density"):
            phys.plot_observable(results, out / "physical_21cm.png")
            phys.plot_eft(results, out / "physical_eft.png")
        print(f"wrote {out}")
    results = load(a.suite / "power_spectrum" / "ps_results.npz", rename)
    out = a.out / "power_spectrum"
    out.mkdir(parents=True, exist_ok=True)
    ps.plot_overlay(results, out / "ps_overlay.png")
    ps.plot_cylindrical_maps(results, out / "ps_cyl_ratio.png", out / "ps_cyl_r.png")
    ps.plot_stage_curves(results, out / "ps_stage_curves.png")
    print(f"wrote {out}")
    results = load(a.suite / "bubble_size" / "bubble_size_results.npz", rename)
    out = a.out / "bubble_size"
    out.mkdir(parents=True, exist_ok=True)
    bubble.plot_distributions(results, out / "bubble_size_distribution.png")
    bubble.plot_summary(results, out / "bubble_size_summary.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
