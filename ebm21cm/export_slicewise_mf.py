"""Reassemble x_HI + T_b cones from the multi-field 2-D slice-wise rhizome.

    python -m ebm21cm.export_slicewise_mf --run runs/rhizome_mf2d_w64_structured \\
        --reference $CUBES/rhizome3d_mf_C/manifest.json --tag rhizome2d_mf_w64 --out-dir $CUBES

For every cone and slice of a reference multi-field ``eval_cubes`` manifest
(``export_cubes3d`` format), the slice's inputs are built exactly as
``ebm21cm.data.mf_slices`` builds its cache rows (fno-21cm windows around the
slice: LOS bands of density and velocity, history band means, the structured
head's physical factor), the 2-D model predicts x_HI and T_b, and the slices are
restacked. Truth is taken from the reference npz and checked against the
window targets. ``--native-cones`` also writes every native slice of those
cones to ``<tag>/native/cone_<id>.h5`` for fno-21cm's per-cone pages.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch

from .data.cache import band_extent
from .data.mf_slices import FNO_ROOT, PREPARATION, band_means, fno_dataset
from .model.rhizome import RhizomeOperator2d
from .train_rhizome_mf2d import tb_normalized


def slice_inputs(dataset, row, idx, head, LOSWindowConfig, bands, ip, chunk=512):
    """cond, scalars, phys, x_HI, normalized T_b for sorted native indices ``idx``
    (same arithmetic as ``mf_slices.build``)."""
    below, above = band_extent(bands)
    halo = max(below, above + 1)
    config = LOSWindowConfig(mode="contiguous", size=chunk + 2 * halo, halo=halo, windows_per_cone=1)
    parts = {k: [] for k in ("cond", "scalars", "phys", "xhi", "tb")}
    for start in range(0, len(dataset.redshifts[row]), chunk):
        sel = idx[(idx >= start) & (idx < start + chunk)]
        if sel.size == 0:
            continue
        sample = dataset.window(row, start - halo, config)
        x, y = sample["x"], sample["y"]
        with torch.no_grad():
            phys = head.physical_factor(x[None])[0].numpy()
        x = x.numpy()
        j = sel - start + halo
        parts["cond"].append(np.concatenate([band_means(x[0], j, bands), band_means(x[1], j, bands)], 1))
        hist = np.concatenate([band_means(x[-2, 0, 0], j, bands), band_means(x[-1, 0, 0], j, bands)], 1)
        parts["scalars"].append(np.concatenate([x[2, 0, 0, j][:, None], x[ip, 0, 0][:, j].T, hist], 1))
        parts["phys"].append(np.moveaxis(phys[..., j], -1, 0))
        parts["xhi"].append(np.moveaxis(y[0].numpy()[..., j], -1, 0))
        parts["tb"].append(np.moveaxis(y[1].numpy()[..., j], -1, 0))
    return {k: np.concatenate(v) for k, v in parts.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True, help="multi-field eval_cubes manifest.json to mirror")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--preparation", default=PREPARATION, help="evaluation preparation (default: cleaned multi-field)")
    ap.add_argument("--native-cones", type=int, nargs="*", default=[])
    ap.add_argument("--native-only", action="store_true", help="write only the --native-cones h5 files")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-cones", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    meta = json.loads((args.run / "metadata.json").read_text())
    attrs = meta["cache_attrs"]
    bands = [tuple(b) for b in json.loads(attrs["bands"])]
    # Evaluate on the cleaned-T_b multi-field preparation (the reference cubes' data), whatever
    # preparation the training cache came from: inputs are identical, raw files lack the T_b cleaning.
    dataset, rows, LOSWindowConfig, head, _, param_names = fno_dataset(args.preparation, FNO_ROOT, attrs["history"])
    ip = [list(dataset.channel_names).index(p) for p in param_names]
    best = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model = RhizomeOperator2d(**best["model_config"]).to(device).eval()
    model.load_state_dict(best["model"])
    tb_off, tb_scale = best["tb_norm"]
    # Sample ids come from the file names (mirror row != sample id after row 72), as in export_cubes3d.
    row_of = {int(re.findall(r"\d+", Path(dataset.file_paths[r]).stem)[-1]): int(r) for r in rows["test"]}

    def predict(inp):
        n = len(inp["cond"])
        xhi = np.empty((n, *inp["cond"].shape[-2:]), np.float32)
        tb = np.empty_like(xhi)
        with torch.no_grad():
            for a in range(0, n, args.batch_size):
                b = slice(a, a + args.batch_size)
                out = model(torch.from_numpy(inp["cond"][b]).to(device), torch.from_numpy(inp["scalars"][b]).to(device))
                t = tb_normalized(out, torch.from_numpy(inp["phys"][b][:, None]).to(device), best["head"], tb_off, tb_scale)
                xhi[b] = out[:, 0].sigmoid().float().cpu().numpy()
                tb[b] = (t[:, 0] * tb_scale + tb_off).float().cpu().numpy()
        return np.moveaxis(xhi, 0, -1), np.moveaxis(tb, 0, -1)

    out = args.out_dir / args.tag
    out.mkdir(parents=True, exist_ok=True)
    for sid in args.native_cones:
        row = row_of[sid]
        z = np.asarray(dataset.redshifts[row])
        inp = slice_inputs(dataset, row, np.arange(len(z)), head, LOSWindowConfig, bands, ip)
        xhi, tb = predict(inp)
        (out / "native").mkdir(exist_ok=True)
        truth_tb = np.moveaxis(inp["tb"], 0, -1) * tb_scale + tb_off
        with h5py.File(out / "native" / f"cone_{sid}.h5", "w") as f:
            for group, field, data in (("target", "neutral_fraction", np.moveaxis(inp["xhi"], 0, -1)),
                                       ("prediction", "neutral_fraction", xhi),
                                       ("target", "brightness_temp", truth_tb), ("prediction", "brightness_temp", tb)):
                f.create_dataset(f"{group}/{field}", data=data.astype(np.float32), compression="gzip", compression_opts=4)
            f["target/neutral_fraction"].attrs["units"] = ""
            f["target/brightness_temp"].attrs["units"] = "mK"
            f["target_z"] = z
            f["lightcone_distances"] = np.asarray(dataset.distances[row])
            f.attrs.update({"cone_id": sid, "row": row, "checkpoint": str((args.run / "best.pt").resolve()),
                            "axis_order": "increasing_redshift", "tag": args.tag,
                            "mapping": json.dumps({"inputs": ["density", "los_velocity"],
                                                   "targets": ["neutral_fraction", "brightness_temp"]}),
                            "sampling": json.dumps({"mode": "slicewise_2d", "bands": bands})})
        print(f"native cone {sid}: {len(z)} slices, x_HI rmse {np.sqrt(np.mean((xhi - np.moveaxis(inp['xhi'], 0, -1)) ** 2)):.4f}",
              flush=True)

    if args.native_only:
        return
    ref = json.loads(args.reference.read_text())
    (ref_tag, entries), = ref["models"].items()
    entries = entries[:args.max_cones]
    written = []
    for k, entry in enumerate(entries):
        sid = int(entry["cone_id"])
        row = int(entry["row"])
        if row_of.get(sid) != row:
            raise ValueError(f"cone {sid}: reference row {row} is not this sample's test row")
        with np.load(entry["npz"]) as d:
            ref_d = {key: d[key] for key in ("truth", "tb_truth", "density", "z_native")}
        z = np.asarray(dataset.redshifts[row])
        idx = np.abs(z[None, :] - ref_d["z_native"][:, None]).argmin(axis=1)
        if np.max(np.abs(z[idx] - ref_d["z_native"])) > 1e-6:
            raise ValueError(f"cone {sid}: reference redshifts are not native slices")
        order = np.argsort(idx, kind="stable")
        if np.any(np.diff(idx[order]) == 0):
            raise ValueError(f"cone {sid}: repeated native slices in the reference grid")
        inp = slice_inputs(dataset, row, idx[order], head, LOSWindowConfig, bands, ip)
        xhi, tb = predict(inp)
        back = np.argsort(order)
        xhi, tb = xhi[..., back], tb[..., back]
        truth = np.moveaxis(inp["xhi"], 0, -1)[..., back]
        truth_tb = (np.moveaxis(inp["tb"], 0, -1) * tb_scale + tb_off)[..., back]
        if not (np.allclose(truth, ref_d["truth"], atol=1e-5) and np.allclose(truth_tb, ref_d["tb_truth"], atol=1e-2)):
            raise ValueError(f"cone {sid}: window targets differ from the reference truth")
        target = out / f"cone_{sid:06d}.npz"
        np.savez(target, pred=xhi, truth=ref_d["truth"], tb_pred=tb.astype(np.float32), tb_truth=ref_d["tb_truth"],
                 density=ref_d["density"], z_native=ref_d["z_native"])
        written.append({**entry, "npz": str(target.resolve())})
        print(f"[{k + 1}/{len(entries)}] sample {sid}: x_HI rmse {np.sqrt(np.mean((xhi - ref_d['truth']) ** 2)):.4f} "
              f"T_b rmse {np.sqrt(np.mean((tb - ref_d['tb_truth']) ** 2)):.2f} mK", flush=True)
    manifest = {"z_grid": ref["z_grid"], "models": {args.tag: written},
                "checkpoint": str((args.run / "best.pt").resolve()), "epoch": best.get("step"), "split": ref.get("split"),
                "selection": f"slice-wise 2-D prediction at the slices of {ref_tag}", "fields": ref.get("fields", {})}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(written)} cones and {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
