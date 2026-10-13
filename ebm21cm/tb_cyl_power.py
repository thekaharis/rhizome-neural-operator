"""Cylindrical (k_perp, k_par) power ratio and coherence of the T_b field, from exported test cubes.

    python -m ebm21cm.tb_cyl_power --manifest runs/eval3d/<suite>/manifest.json \\
        --model rhizome2d_mf_w64_all --out runs/eval3d/tb_cyl/rhizome2d_mf_w64_all.npz

fno-21cm's power-spectrum evaluation computes the cylindrical maps for x_HI only.
This applies its own machinery (``KBinner``, ``build_chunks``, the accumulator's
``_cylindrical_sums`` and ``_safe_ratio_r``) to ``tb_pred`` / ``tb_truth`` of every
cone: per-slice mean removed, Hann-windowed 32-cell LOS chunks centred on
``--chunk-z``, per-cone ratio P_pred/P_true and r, chunks with ~uniform truth
(transverse std < 1e-3 mK) masked, then the median over cones. The output npz uses
the suite's key layout (``<model>/cyl_ratio_med`` etc.), so ``ebm21cm.plot_cyl_error``
reads it directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

FNO_ROOT = "/pfs/10/work/hd_id260-fno_training/fno-21cm"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk-z", type=float, nargs="+", default=[7.0, 9.0, 11.0])
    ap.add_argument("--field", choices=["tb", "xhi"], default="tb",
                    help="xhi reproduces the suite's own x_HI maps (a consistency check)")
    a = ap.parse_args(argv)
    sys.path.insert(0, FNO_ROOT)
    from viz.power_spectrum_evaluation import (KBinner, PowerSpectrumAccumulator, SpectrumConfig, _safe_ratio_r,
                                               build_chunks)

    spec = json.loads(a.manifest.read_text())
    entries = spec["models"][a.model]
    cfg = SpectrumConfig(chunk_z_centers=tuple(a.chunk_z))
    acc = chunks = None
    per_cone = []
    for i, e in enumerate(entries):
        with np.load(e["npz"]) as d:
            keys = ("tb_pred", "tb_truth") if a.field == "tb" else ("pred", "truth")
            pred, truth = (d[k].astype(np.float64) for k in keys)
        if acc is None:
            chunks = build_chunks(cfg, np.asarray(spec["z_grid"]), truth.shape[2])
            acc = PowerSpectrumAccumulator(cfg, KBinner(truth.shape[0], truth.shape[1], cfg.box_mpc, cfg.n_bins),
                                           chunks=chunks)
        rec = []
        for ch in chunks:
            tc, pc = truth[:, :, ch.start:ch.stop], pred[:, :, ch.start:ch.stop]
            ok = bool(tc.std(axis=(0, 1)).mean() > cfg.min_truth_std)
            pt, pp, cross = acc._cylindrical_sums(tc, pc, ch)
            ratio, r = _safe_ratio_r(pt, pp, cross)
            if not ok:
                ratio[:], r[:] = np.nan, np.nan
            rec.append((ratio, r))
        per_cone.append(rec)
        if (i + 1) % 50 == 0:
            print(f"[{a.model}] {i + 1}/{len(entries)} cones", flush=True)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ratio_med = np.stack([np.nanmedian(np.stack([c[j][0] for c in per_cone]), axis=0) for j in range(len(chunks))])
        r_med = np.stack([np.nanmedian(np.stack([c[j][1] for c in per_cone]), axis=0) for j in range(len(chunks))])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    m = a.model
    np.savez_compressed(a.out, **{f"{m}/cyl_ratio_med": ratio_med, f"{m}/cyl_r_med": r_med,
                                  f"{m}/cyl_z_centers": np.array([c.z_center for c in chunks]),
                                  f"{m}/cyl_kpar_edges": np.stack([c.kpar_edges for c in chunks]),
                                  f"{m}/k_edges": acc.binner.edges, f"{m}/n_cones": np.array(len(per_cone))})
    print(f"[{m}] {len(per_cone)} cones -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
