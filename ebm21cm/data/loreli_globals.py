"""LoReLi II per-simulation global histories: the training table of the history emulator.

Sources (all under ``/pfs/10/work/hd_id260-fno_training/data``):

* ``loreli_globals/loreli2_parameters.json`` - grid parameters of every simulation number
  (rule verified on all 9,127 on-grid public simulations) and its public status;
* ``loreli_globals/diagnostics/simuN/diagnostics.dat`` - 21ssd box diagnostics: z and the
  volume-weighted ionized fraction at ~160 snapshots (first two of seven columns);
* ``loreli_globals/powerspectra/simuN/z_dtb_tneutre_xa.txt`` - 21ssd global dTb at ~47 snapshots;
* ``loreli_lc128/lightcone_dtb_no_rsd_simuN.dat`` - (128, 128, 1536) dTb lightcones (7,942);
* ``loreli_lc128/lightcone_xa_tpHI_xion_dens_dtb_no_rsd_simuN.dat`` - five-field cones (228).

Everything is put on LoReLi's native line of sight and on a node grid of ``NODE_BIN``-cell bin
means (192 nodes), which is what the emulator predicts.

**Line-of-sight axis: uniform in redshift, z = 5..15 (1536 planes, dz = 10/1535)**
(:func:`ebm21cm.data.loreli.loreli_redshifts`; it was first assumed uniform in comoving distance).  Verified on the 228
five-field cones against the 21ssd box diagnostics: slice-mean vs box x_HI RMS 0.016 (uniform z)
vs 0.19 (comoving) vs 0.25 (frequency); z(index) recovered by inverting each cone's box x_HI(z)
agrees with uniform z to 0.07 RMS (comoving 1.0); 21ssd global dTb vs slice-mean dTb 3.5 vs 28 mK;
and the LOS lag-1 density correlation rises from 0.60 to 0.71 along the cone as the planes get
closer (2.1 -> 0.85 Mpc).  ``distance_native`` is the true comoving distance of each plane.

Quantities, per simulation:

``xhi_box``    1 - x_ion of the box diagnostics, interpolated in z; NaN below the redshift at
               which the simulation stopped.
``tb_slice``   transverse mean of each lightcone plane [mK] - the quantity the fine-tuning
               cache's T_b history is built from.  Valid planes: the longest run of planes
               that are finite everywhere, minus one partly filled edge plane at each end
               (reproduces ``LoreliCone.valid`` to within a plane), and with no cell beyond
               ``TB_ABS_MAX`` (corrupt planes, e.g. cone 13221 holds 1e12 mK).
``xhi_slice``  transverse-mean x_HI of the five-field cones, on ``LoreliCone.valid`` - the
               cache's x_HI history exactly; ``tb_slice5`` is the cache's T_b history exactly.

Build (CPU; ~760 GB of lightcones, one read each)::

    python -m ebm21cm.data.loreli_globals build --out .../data/compressed/loreli_globals_v1.h5
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .loreli import LORELI_N_LOS, LoreliCone, loreli_redshifts

DATA = Path("/pfs/10/work/hd_id260-fno_training/data")
PARAMS = DATA / "loreli_globals" / "loreli2_parameters.json"
DIAG = DATA / "loreli_globals" / "diagnostics"
GLOBAL_TB = DATA / "loreli_globals" / "powerspectra"
LIGHTCONES = DATA / "loreli_lc128"
SCHEMA = "ebm21cm-loreli-globals-v1"
NAMES = ("f_x", "tau", "rHS", "logMmin", "fesc")
STATUS = {"published": 0, "malformed_file": 1, "not_public": 2, "offgrid_label": 3}
NODE_BIN = 8
TB_ABS_MAX = 1.0e3          # mK; physical |dTb| stays below ~300 mK
MAX_DIAG, MAX_GTB = 400, 64


def los_axis(n: int = LORELI_N_LOS, z_range=(5.0, 15.0)):
    """(z, true comoving distance from z_min) of LoReLi's native LOS planes: uniform in z."""
    return loreli_redshifts(z_range, n)


def grid_params(sim: int, spec: dict) -> tuple:
    """Grid parameters of a simulation number from the rule stored in the parameter file."""
    j = int(sim) - 10000
    tau, logm = spec["PAIRS"][(j // 78) % 42]
    return (spec["FX"][(j // 6) % 13], tau, spec["RH"][j % 6], logm, spec["FE"][j // 3276])


def bin_mean(values: np.ndarray, valid: np.ndarray, size: int = NODE_BIN, min_valid: int | None = None) -> np.ndarray:
    """(n,) -> (n // size,) means over the valid planes of each bin; NaN if fewer than ``min_valid``.

    ``min_valid`` defaults to half a bin, so the end nodes, whose bins always contain the trimmed edge
    plane, still get a value (requiring full bins left them untrained and the emulator unconstrained).
    """
    min_valid = size // 2 if min_valid is None else min_valid
    v = np.asarray(values, np.float64).reshape(-1, size)
    ok = np.asarray(valid, bool).reshape(-1, size) & np.isfinite(v)
    count = ok.sum(1)
    total = np.where(ok, v, 0.0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count >= min_valid, total / count, np.nan).astype(np.float32)


def plane_run(plane_ok: np.ndarray, trim: int = 1) -> np.ndarray:
    """Longest contiguous run of good planes, minus ``trim`` edge planes at each end."""
    idx = np.flatnonzero(plane_ok)
    keep = np.zeros(len(plane_ok), bool)
    if idx.size == 0:
        return keep
    run = max(np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1), key=len)
    if len(run) > 2 * trim:
        keep[run[trim:len(run) - trim]] = True
    return keep


def read_table(path: Path, ncols: int) -> np.ndarray:
    rows = []
    for line in open(path):
        parts = line.split()
        if len(parts) >= ncols:
            try:
                rows.append([float(p) for p in parts[:ncols]])
            except ValueError:
                continue
    return np.asarray(rows, np.float64).reshape(-1, ncols)


def _one(task):
    sim, kind, z_native = task
    n = len(z_native)
    out = {"sim": sim}
    nan = np.full(n, np.nan, np.float32)
    # --- box diagnostics: x_HI(z)
    out["diag"] = np.full((2, MAX_DIAG), np.nan)
    out["xhi_box_native"] = nan.copy()
    p = DIAG / f"simu{sim}" / "diagnostics.dat"
    if p.exists():
        d = read_table(p, 2)
        d = d[np.isfinite(d).all(1)]
        if len(d):
            d = d[np.argsort(d[:, 0])]
            m = min(len(d), MAX_DIAG)
            out["diag"][:, :m] = d[:m].T
            inside = (z_native >= d[0, 0]) & (z_native <= d[-1, 0])
            out["xhi_box_native"] = np.where(inside, 1.0 - np.interp(z_native, d[:, 0], d[:, 1]),
                                             np.nan).astype(np.float32)
    # --- 21ssd global dTb
    out["gtb"] = np.full((2, MAX_GTB), np.nan)
    p = GLOBAL_TB / f"simu{sim}" / "z_dtb_tneutre_xa.txt"
    if p.exists():
        g = read_table(p, 2)
        g = g[np.argsort(g[:, 0])][:MAX_GTB]
        out["gtb"][:, :len(g)] = g.T
    # --- lightcone dTb slice means
    out["tb_native"], out["tb_valid"] = nan.copy(), np.zeros(n, bool)
    out["tb_bad_planes"] = -1
    f = LIGHTCONES / f"lightcone_dtb_no_rsd_simu{sim}.dat"
    if kind >= 1 and f.exists():
        a = np.fromfile(f, dtype="<f4").reshape(128, 128, n)
        finite = np.isfinite(a).all((0, 1))
        absmax = np.where(np.isfinite(a), np.abs(a), 0).max((0, 1))
        geometry = plane_run(finite)
        sane = absmax < TB_ABS_MAX
        valid = geometry & sane
        mean = a.mean((0, 1), dtype=np.float64)
        out["tb_native"] = np.where(valid, mean, np.nan).astype(np.float32)
        out["tb_valid"] = valid
        out["tb_bad_planes"] = int((geometry & ~sane).sum())
        del a
    # --- five-field: the cache's own histories
    out["xhi5_native"], out["tb5_native"], out["valid5"] = nan.copy(), nan.copy(), np.zeros(n, bool)
    out["valid_mismatch"] = -1
    if kind == 2:
        cone = LoreliCone(LIGHTCONES / f"lightcone_xa_tpHI_xion_dens_dtb_no_rsd_simu{sim}.dat",
                          l_perp=294.0)
        with np.errstate(invalid="ignore"):
            xh, tb = cone.history()
        out["valid5"] = cone.valid.copy()
        out["xhi5_native"] = np.where(cone.valid, xh, np.nan).astype(np.float32)
        out["tb5_native"] = np.where(cone.valid, tb, np.nan).astype(np.float32)
        # planes where the dTb-only validity rule disagrees with LoreliCone.valid
        out["valid_mismatch"] = int((cone.valid != out["tb_valid"]).sum())
        del cone
    return out


def build(out: str | Path, workers: int = 32, limit: int | None = None, log=print) -> dict:
    out = Path(out)
    spec = json.load(open(PARAMS))
    sims_info = spec["sims"]
    z_native, distance = los_axis()
    tasks = []
    for key, info in sorted(sims_info.items(), key=lambda kv: int(kv[0])):
        sim = int(key)
        kind = {None: 0, "dtb": 1, "five-field": 2}[info.get("lightcone")]
        if info.get("has_diagnostics") or kind:
            tasks.append((sim, kind, z_native))
    if limit:
        tasks = tasks[:: max(1, len(tasks) // int(limit))][: int(limit)]
    log(f"{len(tasks)} simulations ({sum(t[1] >= 1 for t in tasks)} lightcones, "
        f"{sum(t[1] == 2 for t in tasks)} five-field), {workers} workers")
    t0 = time.time()
    results = []
    with mp.get_context("fork").Pool(workers) as pool:
        for k, r in enumerate(pool.imap(_one, tasks, chunksize=4)):
            results.append(r)
            if (k + 1) % 500 == 0:
                log(f"  {k + 1}/{len(tasks)}  {time.time() - t0:.0f} s")
    nsim, nn = len(results), LORELI_N_LOS // NODE_BIN
    sims = np.array([r["sim"] for r in results], np.int32)
    stack = lambda key, dt: np.stack([r[key] for r in results]).astype(dt)
    xhi_box_native = stack("xhi_box_native", np.float32)
    tb_native, tb_valid = stack("tb_native", np.float32), stack("tb_valid", bool)
    xhi5, tb5, valid5 = stack("xhi5_native", np.float32), stack("tb5_native", np.float32), stack("valid5", bool)
    box_valid = np.isfinite(xhi_box_native)
    nodes = lambda v, ok: np.stack([bin_mean(v[i], ok[i]) for i in range(nsim)])
    distance_node = distance.reshape(-1, NODE_BIN).mean(1)
    z_node = np.interp(distance_node, distance, z_native)
    theta = np.array([grid_params(s, spec) for s in sims], np.float64)
    status = np.array([STATUS[sims_info[str(s)]["status"]] for s in sims], np.int8)
    kind = np.array([t[1] for t in tasks], np.int8)
    tb_bad = np.array([r["tb_bad_planes"] for r in results], np.int16)
    mismatch = np.array([r["valid_mismatch"] for r in results], np.int16)
    tmp = out.with_name(out.name + ".tmp")
    with h5py.File(tmp, "w") as h:
        put = lambda name, data: h.create_dataset(name, data=data, compression="lzf")
        put("sim", sims); put("theta", theta); put("status", status); put("lightcone", kind)
        put("block", (sims - 10000) // 6)
        put("z_native", z_native); put("distance_native", distance)
        put("z_node", z_node); put("distance_node", distance_node)
        put("xhi_box_native", xhi_box_native); put("xhi_box", nodes(xhi_box_native, box_valid))
        put("tb_native", tb_native); put("tb_valid", tb_valid); put("tb_slice", nodes(tb_native, tb_valid))
        put("tb_bad_planes", tb_bad)
        put("xhi5_native", xhi5); put("tb5_native", tb5); put("valid5", valid5)
        put("xhi_slice", nodes(xhi5, valid5)); put("tb_slice5", nodes(tb5, valid5))
        put("valid5_mismatch", mismatch)
        put("diag", stack("diag", np.float64)); put("global_tb", stack("gtb", np.float64))
        h.attrs.update({
            "schema": SCHEMA, "created": datetime.now(timezone.utc).isoformat(),
            "param_names": json.dumps(NAMES), "status_codes": json.dumps(STATUS),
            "node_bin": NODE_BIN, "n_nodes": nn, "tb_abs_max_mK": TB_ABS_MAX,
            "lightcone_kind": "0 none, 1 dTb only, 2 five-field",
            "los_axis": "uniform in z, 5..15, 1536 planes; distance_native = true comoving distance",
            "diag_columns": "z, volume-weighted x_ion (rows sorted by z, NaN padded)",
            "global_tb_columns": "z, global dTb [mK] from 21ssd z_dtb_tneutre_xa.txt",
            "params_file": str(PARAMS), "params_rule": spec["rule"], "sources": json.dumps(
                [str(DIAG), str(GLOBAL_TB), str(LIGHTCONES)]),
        })
    tmp.replace(out)
    five = kind == 2
    summary = {
        "simulations": int(nsim), "with_box_x_HI": int(box_valid.any(1).sum()),
        "with_lightcone_T_b": int(tb_valid.any(1).sum()), "five_field": int(five.sum()),
        "tb_bad_planes": {str(int(s)): int(b) for s, b in zip(sims, tb_bad) if b > 0},
        "five_field_valid_mismatch_planes": {str(int(s)): int(m) for s, m in zip(sims, mismatch) if m > 0},
        "seconds": round(time.time() - t0),
    }
    log(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ebm21cm.data.loreli_globals")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=32)
    b.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    build(a.out, a.workers, a.limit, log=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
