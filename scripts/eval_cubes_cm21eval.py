"""21cmEVAL manifests for the cones of an ``eval_cubes`` export (21cmFAST test set), in chunks.

    python scripts/eval_cubes_cm21eval.py --cubes $CUBES/rhizome2d_mf_w64_all --out $CUBES/21cmfast_test_eval21 \
        --chunk 25 [--check $CUBES/mf_cnn_whno_es_tbs_histtb_tbclean_ep26]

Writes ``<out>/<chunk>/manifest.json`` (target: the true x_HI array ``truth``, input ``density``) and
``manifest_tb.json`` (target ``tb_truth``, inputs ``density`` and ``truth``, so a T_b pass is staged by the
true x_HI with ``--set stage_field=truth``). Samples point at the existing cube files (absolute paths),
nothing is copied; per-cone Omega_m comes from the export manifest, the box is ``--box-mpc``. Each chunk
is one 21cmEVAL "group" (``eval21 plot`` pools groups exactly). ``--check`` verifies that another export
covers the same cones on the same grid, with identical truth on the first cone.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cubes", type=Path, required=True, help="eval_cubes dir with manifest.json (truth source)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--box-mpc", type=float, default=200.0)
    ap.add_argument("--check", type=Path, nargs="*", default=[])
    args = ap.parse_args()

    man = json.loads((args.cubes / "manifest.json").read_text())
    (entries,) = man["models"].values()
    z = [float(v) for v in man["z_grid"]]
    for other in args.check:
        om = json.loads((other / "manifest.json").read_text())
        (oe,) = om["models"].values()
        assert [e["cone_id"] for e in oe] == [e["cone_id"] for e in entries], f"{other}: different cones"
        assert np.allclose(om["z_grid"], z), f"{other}: different z grid"
        with np.load(entries[0]["npz"]) as a, np.load(oe[0]["npz"]) as b:
            assert np.array_equal(a["truth"], b["truth"]) and np.array_equal(a["tb_truth"], b["tb_truth"]), \
                f"{other}: truth differs"
    samples = [{"id": Path(e["npz"]).stem, "path": str(Path(e["npz"]).resolve()), "split": "test",
                "cone_id": int(e["cone_id"]), "omega_m": float(e["omega_m"])} for e in entries]
    chunks = [samples[i:i + args.chunk] for i in range(0, len(samples), args.chunk)]
    for k, chunk in enumerate(chunks):
        d = args.out / f"C{k:02d}"
        d.mkdir(parents=True, exist_ok=True)
        common = {"box_mpc": args.box_mpc, "z_grid": z, "samples": chunk}
        (d / "manifest.json").write_text(json.dumps({**common, "target": "truth", "inputs": ["density"]}, indent=1))
        (d / "manifest_tb.json").write_text(json.dumps({**common, "target": "tb_truth",
                                                        "inputs": ["density", "truth"]}, indent=1))
    print(f"{len(samples)} cones in {len(chunks)} chunks under {args.out}")


if __name__ == "__main__":
    main()
