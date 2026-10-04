"""Reassemble 3-D x_HI cones from a 2-D slice model, one transverse slice at a time.

    python -m ebm21cm.export_slicewise --run runs/rhizome_xhi_w128 \\
        --reference $WORK/data/eval_cubes/rhizome3d_w48_cont10/manifest.json \\
        --split test --tag rhizome2d_w128_slicewise --out-dir $WORK/data/eval_cubes

Every slice of a reference eval_cubes manifest (``export_cubes3d`` format: the
nearest native slice per grid redshift) is predicted independently by a
``train_rhizome`` model from its own LOS density bands, built from the raw
lightcone exactly as ``ebm21cm.data.cache`` builds them. ``truth``, ``density``,
``z_native`` and the cone order come from the reference npz files, so the
result merges with the reference models in ``eval3d_suite.sbatch``.

Only cones in the 2-D cache's ``--split`` are exported (``--split all`` disables
the filter): the cache's own 80/10/10 split differs from fno-21cm's, so most
3-D test cones are 2-D *training* cones. Bands that would reach past either end
of the cone use the end slices repeatedly (edge padding); the count of such
slices is recorded in the manifest.

``--native-cones`` additionally predicts *every* native slice of those cones and
writes ``<tag>/native/cone_<id>.h5`` in ``export_cubes3d``'s format, for
fno-21cm's per-cone pages; ``--native-only`` writes just these.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .data.cache import SPLITS, Normalizer, band_extent, read_stats
from .data.lightcone import Lightcone
from .model.rhizome import RhizomeOperator2d


def band_stack(density: np.ndarray, centers: np.ndarray, bands) -> tuple[np.ndarray, int]:
    """(len(centers), C, H, W) band means of an (n_los, H, W) density cube, with edge
    padding; also the number of centers whose bands needed padding."""
    below, above = band_extent(bands)
    padded = np.concatenate([np.repeat(density[:1], below, 0), density, np.repeat(density[-1:], above, 0)])
    csum = np.concatenate([np.zeros((1, *density.shape[1:]), np.float64), np.cumsum(padded, 0, dtype=np.float64)])
    c = centers + below
    out = np.stack([(csum[c + hi] - csum[c + lo]) / (hi - lo) for lo, hi in bands], axis=1)
    edge = int(np.sum((centers < below) | (centers + above >= len(density))))
    return out.astype(np.float32), edge


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="train_rhizome run directory (best.pt)")
    ap.add_argument("--reference", type=Path, required=True, help="eval_cubes manifest.json to mirror")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split", choices=("train", "validation", "test", "eval", "all"), default="test",
                    help="2-D cache split of the exported cones (eval = validation + test)")
    ap.add_argument("--raw-dir", type=Path, default=Path("/pfs/10/work/hd_id260-fno_training/data/data"))
    ap.add_argument("--raw-pattern", default="21cmfast_11d_sample{:06d}.h5")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-cones", type=int, default=None)
    ap.add_argument("--native-cones", type=int, nargs="*", default=[], help="sample ids to also write at native resolution")
    ap.add_argument("--native-only", action="store_true", help="write only the --native-cones h5 files")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    meta = json.loads((args.run / "metadata.json").read_text())
    cache_path = meta["cache"]
    stats = read_stats(cache_path)
    norm = Normalizer(stats)
    with h5py.File(cache_path, "r") as h:
        bands = [tuple(b) for b in json.loads(h.attrs["bands"])]
        param_names = json.loads(h.attrs["param_names"])
        split_of = {int(k): v for k, v in json.loads(h.attrs["split_of"]).items()}
    channels = meta.get("channels") or list(range(len(bands)))
    if isinstance(channels, str):
        channels = json.loads(channels)
    wanted = {"all": None, "eval": {1, 2}}.get(args.split, {SPLITS.get(args.split)})

    device = torch.device(args.device)
    best = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    model = RhizomeOperator2d(**meta["model_config"]).to(device).eval()
    model.load_state_dict(best["model"])
    mean, std = float(stats["delta_log1p_mean"]), float(stats["delta_log1p_std"])

    def predict(density, idx, z, params):
        cond, edge = band_stack(density, idx, bands)
        cond = cond[:, channels]
        scalars = norm.scalars(z[idx], np.tile(params, (len(idx), 1))).astype(np.float32)
        pred = np.empty((len(idx), *density.shape[1:]), np.float32)
        with torch.no_grad():
            for a in range(0, len(idx), args.batch_size):
                c = torch.from_numpy(cond[a:a + args.batch_size]).to(device)
                c = (torch.log1p(c.clamp_min(-0.99)) - mean) / std
                s = torch.from_numpy(scalars[a:a + args.batch_size]).to(device)
                pred[a:a + len(c)] = model(c, s).sigmoid()[:, 0].float().cpu().numpy()
        return np.moveaxis(pred, 0, -1), edge

    out = args.out_dir / args.tag
    for sid in args.native_cones:
        with Lightcone(args.raw_dir / args.raw_pattern.format(sid)) as lc:
            z, dist = np.asarray(lc.redshifts), np.asarray(lc.distances)
            density = lc.read_range("density", 0, lc.n_los)
            xhi = np.moveaxis(lc.read_range("neutral_fraction", 0, lc.n_los), 0, -1)
            params = lc.params(param_names, missing="nan")
        pred, edge = predict(density, np.arange(len(z)), z, params)
        (out / "native").mkdir(parents=True, exist_ok=True)
        with h5py.File(out / "native" / f"cone_{sid}.h5", "w") as f:
            f.create_dataset("target/neutral_fraction", data=xhi.astype(np.float32), compression="gzip",
                             compression_opts=4)
            f.create_dataset("prediction/neutral_fraction", data=pred, compression="gzip", compression_opts=4)
            f["target/neutral_fraction"].attrs["units"] = ""
            f["target_z"] = z
            f["lightcone_distances"] = dist
            f.attrs.update({"cone_id": sid, "split_2d": int(split_of.get(sid, -1)),
                            "checkpoint": str((args.run / "best.pt").resolve()),
                            "axis_order": "increasing_redshift", "tag": args.tag,
                            "mapping": json.dumps({"inputs": ["density"], "targets": ["neutral_fraction"]}),
                            "sampling": json.dumps({"mode": "slicewise_2d", "bands": bands})})
        print(f"native cone {sid}: {len(z)} slices, rmse {np.sqrt(np.mean((pred - xhi) ** 2)):.4f}, "
              f"edge-padded {edge}", flush=True)
    if args.native_only:
        return

    ref = json.loads(args.reference.read_text())
    (ref_tag, ref_entries), = ref["models"].items()
    entries = [e for e in ref_entries if wanted is None or split_of.get(int(e["cone_id"])) in wanted]
    entries = entries[:args.max_cones]
    if not entries:
        raise SystemExit("no reference cone falls in the requested 2-D split")
    out.mkdir(parents=True, exist_ok=True)
    written, edge_total = [], 0
    for k, entry in enumerate(entries):
        sid = int(entry["cone_id"])
        with np.load(entry["npz"]) as d:
            truth, density_ref, z_ref = d["truth"], d["density"], d["z_native"]
        with Lightcone(args.raw_dir / args.raw_pattern.format(sid)) as lc:
            z = np.asarray(lc.redshifts)
            idx = np.abs(z[None, :] - z_ref[:, None]).argmin(axis=1)
            if np.max(np.abs(z[idx] - z_ref)) > 1e-6:
                raise ValueError(f"cone {sid}: reference redshifts are not native lightcone slices")
            density = lc.read_range("density", 0, lc.n_los)
            xhi = lc.read_range("neutral_fraction", 0, lc.n_los)
            params = lc.params(param_names, missing="nan")
        # The raw slices must be the reference slices (same axis order and orientation).
        if not np.allclose(np.moveaxis(xhi[idx], 0, -1), truth, atol=1e-5):
            raise ValueError(f"cone {sid}: raw x_HI slices do not match the reference truth")
        pred, edge = predict(density, idx, z, params)
        edge_total += edge
        target = out / f"cone_{sid:06d}.npz"
        np.savez(target, pred=pred, truth=truth, density=density_ref, z_native=z_ref)
        written.append({**entry, "npz": str(target.resolve()), "split_2d": split_of.get(sid)})
        print(f"[{k + 1}/{len(entries)}] sample {sid} (2-D split {split_of.get(sid)}) "
              f"rmse {np.sqrt(np.mean((pred - truth) ** 2)):.4f} edge-padded slices {edge}", flush=True)
    manifest = {"z_grid": ref["z_grid"], "models": {args.tag: written},
                "checkpoint": str((args.run / "best.pt").resolve()), "epoch": best.get("step"),
                "split": f"2-D cache {args.split} cones of reference {ref_tag} ({ref.get('split')})",
                "selection": "slice-wise 2-D prediction at the reference slices", "edge_padded_slices": edge_total,
                "fields": ref.get("fields", {})}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    # Per-split manifests, so the suite can be run on the cleanest subset alone.
    for name, code in (("validation", 1), ("test", 2)):
        part = [e for e in written if e["split_2d"] == code]
        if part and len(part) < len(written):
            (out / f"manifest_{name}.json").write_text(json.dumps({**manifest, "models": {args.tag: part}}, indent=1))
    print(f"wrote {len(written)} cones and {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
