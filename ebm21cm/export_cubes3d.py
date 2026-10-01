"""Export full-cone x_HI predictions of a 3-D rhizome or fno-21cm LOS-window model.

    python -m ebm21cm.export_cubes3d --rhizome runs/rhizome3d/w48_m36_h200 --tag rhizome_w48 \\
        --out-dir $WORK/data/eval_cubes --h5-cones 24 27 32
    python -m ebm21cm.export_cubes3d --fno $FNO/checkpoints/3d_xhi/los_windows/<run>/best.pt \\
        --tag fno_cnn_whno_es --out-dir $WORK/data/eval_cubes

Every cone of the split is predicted at native resolution with fno-21cm's own
tiled inference (``predict_native_cone``) and written in the formats its
evaluation suite reads:

* ``<out-dir>/<tag>/cone_<sample>.npz`` + ``manifest.json``: ``pred``,
  ``truth`` (x_HI), ``density`` and ``z_native`` on a common redshift grid,
  taking the nearest native slice per grid redshift (never interpolating),
  exactly as ``tools_export_physical_cubes.py`` does. For
  ``viz.physical_evaluation``, ``bubble_size_evaluation``,
  ``power_spectrum_evaluation`` and ``boundary_band_diagnostic`` (``--manifest``).
* optional ``<out-dir>/<tag>/native/cone_<sample>.h5`` at full native LOS
  resolution (``target``/``prediction`` groups, ``target_z``) for
  ``viz.multifield_detailed``, ``multifield_lightcone_strips`` and
  ``multifield_slices``.

``cone_id`` is the 21cmFAST sample id parsed from the lightcone file name.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from .train_rhizome3d import fno_pipeline, install_histories


def load_model(args, device):
    """(model, dataset, rows, window, description) for either model family."""
    fm, FieldMapping, FieldRegistry, LOSWindowConfig, _ = fno_pipeline(args.fno_root)
    if args.fno:
        checkpoint, model, dataset, rows = fm.restore(args.fno, device)
        window = LOSWindowConfig(**checkpoint["metadata"]["sampling"])
        return fm, model, dataset, rows, window, {"checkpoint": str(Path(args.fno).resolve()),
                                                  "epoch": checkpoint.get("epoch")}
    from .model.rhizome3d import RhizomeOperator3d

    run = Path(args.rhizome)
    meta = json.loads((run / "metadata.json").read_text())
    preparation = fm.read_json(meta["preparation"])
    registry = FieldRegistry.from_dict(preparation["registry"])
    # Runs predating multi-field support recorded no mapping: density -> x_HI.
    fields = meta.get("mapping") or {"inputs": ["density"], "targets": ["neutral_fraction"]}
    mapping = FieldMapping.create(fields["inputs"], fields["targets"], preparation["conditioning"], registry)
    dataset, rows, _ = fm.prepared_dataset(preparation, mapping)
    install_histories(fm, dataset, meta.get("history_spec") or "", args.fno_root)
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model = RhizomeOperator3d(**best["model_config"]).to(device).eval()
    model.load_state_dict(best["model"])
    window = LOSWindowConfig(**meta["window"])
    return fm, model, dataset, rows, window, {"checkpoint": str((run / "best.pt").resolve()),
                                              "epoch": best["epoch"], "val_subset": best["val"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rhizome", help="train_rhizome3d run directory (uses best.pt)")
    src.add_argument("--fno", help="fno-21cm LOS-window checkpoint (best.pt)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fno-root", default="/pfs/10/work/hd_id260-fno_training/fno-21cm")
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--n-z", type=int, default=512)
    ap.add_argument("--z-min", type=float, default=5.001)
    ap.add_argument("--z-max", type=float, default=24.97)
    ap.add_argument("--max-cones", type=int, default=None)
    ap.add_argument("--h5-cones", type=int, nargs="*", default=[], help="sample ids to also write at native resolution")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    fm, model, dataset, rows, window, description = load_model(args, device)
    # fno-21cm modules are importable only after load_model put its root on sys.path.
    from dataset.los_windows import predict_native_cone
    from dataset.lightcone_params import PARAM_NAMES

    if list(dataset.mapping.targets)[0] != "neutral_fraction":
        raise ValueError("the first target must be neutral_fraction")
    norm = dataset.normalization["neutral_fraction"]
    has_tb = "brightness_temp" in dataset.mapping.targets
    tb_norm = dataset.normalization["brightness_temp"] if has_tb else None
    z_grid = np.linspace(args.z_min, args.z_max, args.n_z)
    out = args.out_dir / args.tag
    (out / "native").mkdir(parents=True, exist_ok=True)
    entries = []
    split_rows = list(rows[args.split])[:args.max_cones]
    for k, row in enumerate(split_rows):
        path = Path(dataset.file_paths[row])
        sid = int(re.findall(r"\d+", path.stem)[-1])
        z = np.asarray(dataset.redshifts[row])
        if z[0] > args.z_min + 1e-3 or z[-1] < args.z_max - 1e-3:
            raise ValueError(f"row {row} does not cover the export grid")
        pick = np.abs(z[None, :] - z_grid[:, None]).argmin(axis=1)
        if len(np.unique(pick)) != len(pick):
            raise ValueError("export grid finer than the native spacing (repeated slices)")
        both = predict_native_cone(model, dataset, row, window, device).numpy()
        pred = both[0] * norm["scale"] + norm["offset"]
        truth = dataset.read_fields(row, names=["neutral_fraction", "density"] + (["brightness_temp"] if has_tb else []))
        target = f"cone_{sid:06d}.npz"
        extra = {}
        if has_tb:  # same keys as fno-21cm's multi-field cubes (mK)
            tb_pred = both[1] * tb_norm["scale"] + tb_norm["offset"]
            extra = {"tb_pred": tb_pred[..., pick].astype(np.float32),
                     "tb_truth": truth["brightness_temp"][..., pick].astype(np.float32)}
        np.savez(out / target, pred=pred[..., pick].astype(np.float32),
                 truth=truth["neutral_fraction"][..., pick].astype(np.float32),
                 density=truth["density"][..., pick].astype(np.float32), z_native=z[pick], **extra)
        omm = float(dataset.params[row][PARAM_NAMES.index("OMm")])
        entries.append({"npz": str((out / target).resolve()), "cone_id": sid, "row": int(row), "omega_m": omm})
        if sid in args.h5_cones:
            import h5py

            with h5py.File(out / "native" / f"cone_{sid}.h5", "w") as f:
                f.create_dataset("target/neutral_fraction", data=truth["neutral_fraction"].astype(np.float32),
                                 compression="gzip", compression_opts=4)
                f.create_dataset("prediction/neutral_fraction", data=pred.astype(np.float32),
                                 compression="gzip", compression_opts=4)
                if has_tb:
                    f.create_dataset("target/brightness_temp", data=truth["brightness_temp"].astype(np.float32),
                                     compression="gzip", compression_opts=4)
                    f.create_dataset("prediction/brightness_temp", data=tb_pred.astype(np.float32),
                                     compression="gzip", compression_opts=4)
                    f["target/brightness_temp"].attrs["units"] = "mK"
                f["target/neutral_fraction"].attrs["units"] = ""
                f["target_z"] = z
                f["lightcone_distances"] = np.asarray(dataset.distances[row])
                f.attrs.update({"cone_id": sid, "row": int(row), "checkpoint": description["checkpoint"],
                                "axis_order": "increasing_redshift", "tag": args.tag,
                                "mapping": json.dumps({"inputs": list(dataset.mapping.inputs),
                                                       "targets": list(dataset.mapping.targets)}),
                                "sampling": json.dumps(window.to_dict())})
        print(f"[{k + 1}/{len(split_rows)}] sample {sid} (row {row}) "
              f"rmse {np.sqrt(np.mean((pred - truth['neutral_fraction']) ** 2)):.4f}", flush=True)
    manifest = {"z_grid": z_grid.tolist(), "models": {args.tag: entries}, **description,
                "split": args.split, "selection": "nearest native slice per grid redshift",
                "fields": {"pred/truth": "neutral_fraction", "density": "overdensity (input)",
                           **({"tb_pred/tb_truth": "brightness_temp [mK]"} if has_tb else {})}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(entries)} cones and {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
