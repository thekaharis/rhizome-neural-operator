"""Export LoReLi v3 test-lightcone predictions as fno-21cm ``eval_cubes`` (+ optional native cylindrical power).

    python scripts/loreli_export_eval_cubes.py --model b8_pointwise \
        [--native-cyl runs/eval3d/native_cyl/loreli_v3_b8_pointwise] [--cyl-z 5.5 6 7 8 9]

The LoReLi counterpart of ``ebm21cm.export_slicewise_mf`` / ``export_cubes3d``: it writes the cone
files and manifests that the repository's viz tools read (``ebm21cm.compare_cones3d``,
``ebm21cm.delta_slices``, ``ebm21cm.tb_cyl_power``) and, with ``--native-cyl``, feeds
``ebm21cm.native_cyl.NativeCyl`` exactly as those exporters do (for ``ebm21cm.plot_cyl_error``).
Nothing is predicted here: the inputs are the 21cmEVAL pipeline's full test lightcones (truth in
``--truth-root/<group>/cone<N>.npz``, predictions in ``--preds/<model>/cone<N>.npz`` from
``scripts/loreli_cm21eval_predict.py``; interior 206 x 206 cells of 294 Mpc, LOS on the 1.4286 Mpc
comoving grid).

The test cones come in geometry groups with different z windows (``runs/cm21eval/groups.json``),
so there is one manifest per group, ``<out>/manifest_<group>.json``, each on one common z grid;
the cone files ``<out>/cone_<id>.npz`` hold ``pred``/``truth`` (x_HI), ``tb_pred``/``tb_truth``
[mK], ``density`` (1 + delta) and ``z_native``, stored as float16.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
W = Path("/pfs/10/work/hd_id260-fno_training")
L_PERP, N_PERP = 294.0, 206


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="model name under --preds (e.g. reference, b8_pointwise)")
    ap.add_argument("--tag", default=None, help="manifest model name (default loreli_v3_<model>)")
    ap.add_argument("--truth-root", type=Path, default=W / "data/eval_cubes/loreli_l294_v3_test")
    ap.add_argument("--preds", type=Path, default=REPO / "runs/cm21eval/preds")
    ap.add_argument("--groups", type=Path, default=REPO / "runs/cm21eval/groups.json")
    ap.add_argument("--out", type=Path, default=None, help="default data/eval_cubes/<tag>")
    ap.add_argument("--native-cyl", default=None, metavar="PREFIX",
                    help="also write native-resolution cylindrical x_HI/T_b power (ebm21cm.native_cyl) to PREFIX_<field>.npz")
    ap.add_argument("--cyl-z", type=float, nargs="+", default=[5.5, 6.0, 7.0, 8.0, 9.0])
    ap.add_argument("--cyl-only", action="store_true", help="only the native cylindrical power, no cubes")
    a = ap.parse_args(argv)
    tag = a.tag or f"loreli_v3_{a.model}"
    out = a.out or W / "data/eval_cubes" / tag
    groups = json.loads(a.groups.read_text())
    prov = json.loads((a.preds / a.model / "provenance.json").read_text())
    cyl = None
    if a.native_cyl:
        from ebm21cm.native_cyl import NativeCyl
        cyl = NativeCyl(a.cyl_z, box_mpc=L_PERP, n_perp=N_PERP)
    if not a.cyl_only:
        out.mkdir(parents=True, exist_ok=True)
    for g, spec in groups.items():
        entries, z_grid = [], None
        for row, c in enumerate(spec["cones"]):
            with np.load(a.truth_root / g / f"cone{c}.npz") as t:
                truth, tb_truth, density, z = t["xhi"], t["tb"], t["density"], t["z_grid"].astype(np.float64)
            with np.load(a.preds / a.model / f"cone{c}.npz") as p:
                if not np.allclose(p["z_grid"], z):
                    raise ValueError(f"{a.model}/cone{c}: prediction z grid differs from the truth")
                pred, tb_pred = p["xhi"], p["tb"]
            if z_grid is None:
                z_grid = z
            elif not np.allclose(z, z_grid):
                raise ValueError(f"group {g}: cone {c} is on a different z grid")
            if not a.cyl_only:
                path = out / f"cone_{c:06d}.npz"
                np.savez(path, pred=pred.astype(np.float16), truth=truth.astype(np.float16),
                         tb_pred=tb_pred.astype(np.float16), tb_truth=tb_truth.astype(np.float16),
                         density=density.astype(np.float16), z_native=z)
                entries.append({"npz": str(path), "cone_id": int(c), "row": row, "omega_m": 0.308, "group": g})
            if cyl is not None:
                cyl.add("xhi", c, z, truth, pred)
                cyl.add("tb", c, z, tb_truth, tb_pred)
            print(f"[{g}] cone {c}", flush=True)
        if not a.cyl_only:
            (out / f"manifest_{g}.json").write_text(json.dumps({
                "z_grid": z_grid.tolist(), "models": {tag: entries}, "split": "test",
                "checkpoint": str(Path(prov["run"]) / "best.pt"),
                "fields": {"pred/truth": "neutral_fraction", "density": "1 + delta (input)",
                           "tb_pred/tb_truth": "brightness_temp [mK]"},
                "selection": (f"LoReLi II v3 test lightcones, geometry group {g} (z {z_grid[0]:.3f}-{z_grid[-1]:.3f}), "
                              f"interior {N_PERP}^2 cells / {L_PERP:g} Mpc, LOS 1.4286 Mpc comoving grid; "
                              f"histories: {prov['histories']}")}, indent=1))
    if cyl is not None:
        print("wrote", *cyl.save(a.native_cyl, tag), flush=True)


if __name__ == "__main__":
    main()
