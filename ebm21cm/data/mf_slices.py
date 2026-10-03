"""Multi-field slice cache for the 2-D slice-wise rhizome, built from fno-21cm's
LOS-window preparation (density + LOS velocity -> x_HI + T_b).

    python -m ebm21cm.data.mf_slices build --out shard000.h5 --shard 0 --n-shards 32
    python -m ebm21cm.data.mf_slices merge --out mf_slices.h5 shard*.h5

Every value comes from fno-21cm's own native-window pipeline (``prepared_dataset``
plus the installed global-history emulators), so inputs, targets, splits and
normalization are exactly those of the 3-D multi-field runs. Per cone, slices are
sampled like ``ebm21cm.data.cache`` (``slices_per_cone`` stratified in x_HI inside
the cone's reionization window, ``background_slices`` outside it), and each row holds:

    cond     (N, 26, H, W)  LOS band means of normalized density (13) and LOS velocity (13)
    phys     (N, H, W)      A(z) (1+delta) V(dv/dr) in mK: the structured T_b head's
                            physical factor (T_b = x_HI * phys * (1 - exp(u))), computed
                            with ``StructuredBrightness3d`` on the surrounding window
    xhi, tb  (N, H, W)      targets: x_HI and normalized T_b, as fno-21cm's targets
    scalars  (N, 38)        1/(1+z), 11 normalized parameters, and LOS band means of the
                            emulated global x_HI and T_b histories (13 each)
    z, los_index, cone_id, split
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from .cache import DEFAULT_BANDS, band_extent, parse_bands, pick_window_indices
from .memory import _runs, dihedral

SCHEMA = "ebm21cm-mfslices-v1"
SPLITS = {"train": 0, "val": 1, "validation": 1, "test": 2}
FNO_ROOT = "/pfs/10/work/hd_id260-fno_training/fno-21cm"
PREPARATION = FNO_ROOT + "/experiments/los_windows/preparation_multifield_2000_tbclean.json"
HISTORY = "experiments/global_history/emulator.pt,experiments/global_history/emulator_brightness_temp.pt"


def fno_dataset(preparation, fno_root, history):
    from ..train_rhizome3d import fno_pipeline, install_histories, structured_config
    from ..model.rhizome3d import StructuredBrightness3d

    fm, FieldMapping, FieldRegistry, LOSWindowConfig, _ = fno_pipeline(fno_root)
    from dataset.lightcone_params import PARAM_NAMES

    prep = fm.read_json(preparation)
    mapping = FieldMapping.create(["density", "los_velocity"], ["neutral_fraction", "brightness_temp"],
                                  prep["conditioning"], FieldRegistry.from_dict(prep["registry"]))
    dataset, rows, _ = fm.prepared_dataset(prep, mapping)
    install_histories(fm, dataset, history, fno_root)
    structured = structured_config(mapping, dataset.normalization, dataset.parameter_normalization, PARAM_NAMES)
    return dataset, rows, LOSWindowConfig, StructuredBrightness3d(**structured), structured, list(PARAM_NAMES)


def band_means(values, centers, bands):
    """(len(centers), *values.shape[:-1]) band means along the last axis; ``centers``
    index ``values`` and every band must lie inside it."""
    csum = np.concatenate([np.zeros((*values.shape[:-1], 1)), np.cumsum(values, -1, dtype=np.float64)], -1)
    return np.stack([np.stack([(csum[..., c + hi] - csum[..., c + lo]) / (hi - lo) for lo, hi in bands])
                     for c in centers]).astype(np.float32)


def build(out, shard=0, n_shards=1, preparation=PREPARATION, fno_root=FNO_ROOT, history=HISTORY,
          slices_per_cone=48, background_slices=6, xhi_window=(0.02, 0.98), bands=DEFAULT_BANDS,
          z_range=(5.001, 24.97), chunk=512, sample_seed=0, limit=None, log=print):
    bands = parse_bands(bands) if isinstance(bands, str) else [tuple(b) for b in bands]
    below, above = band_extent(bands)
    halo = max(below, above + 1)
    dataset, rows, LOSWindowConfig, head, structured, param_names = fno_dataset(preparation, fno_root, history)
    names = list(dataset.channel_names)
    if names[:3] != ["density", "los_velocity", "1/(1+z)"] or names[-2:] != ["x_HI_global_emulated",
                                                                              "T_b_global_emulated"]:
        raise ValueError(f"unexpected fno-21cm channel layout {names}")
    ip = [names.index(p) for p in param_names]
    split_of = {int(r): SPLITS[s] for s, rs in rows.items() for r in rs}
    mine = sorted(split_of)[shard::n_shards][:limit]
    config = LOSWindowConfig(mode="contiguous", size=chunk + 2 * halo, halo=halo, windows_per_cone=1)
    H, W = dataset.transverse_shape
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    n_rows = 0
    with h5py.File(tmp, "w") as h:
        def mk(name, shape, dt):
            return h.create_dataset(name, shape=(0, *shape), maxshape=(None, *shape), dtype=dt,
                                    chunks=(1, *shape) if len(shape) > 1 else (1024, *shape))
        d = {"cond": mk("cond", (2 * len(bands), H, W), np.float16), "phys": mk("phys", (H, W), np.float32),
             "xhi": mk("xhi", (H, W), np.float16), "tb": mk("tb", (H, W), np.float16),
             "scalars": mk("scalars", (1 + len(param_names) + 2 * len(bands),), np.float32),
             "z": mk("z", (), np.float64), "los_index": mk("los_index", (), np.int32),
             "cone_id": mk("cone_id", (), np.int64), "split": mk("split", (), np.int8)}
        for k, row in enumerate(mine):
            cid = int(dataset.cone_ids[row])
            z = np.asarray(dataset.redshifts[row])
            history_xhi = dataset.read_fields(row, names=["neutral_fraction"])["neutral_fraction"].mean((0, 1))
            rng = np.random.default_rng([sample_seed, cid])
            idx = np.sort(pick_window_indices(z, history_xhi, below, above, slices_per_cone, background_slices,
                                              xhi_window, z_range, rng))
            parts = {key: [] for key in d}
            for start in range(0, len(z), chunk):
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
            n = idx.size
            values = {key: np.concatenate(v) for key, v in parts.items() if v}
            values.update({"z": z[idx], "los_index": idx, "cone_id": np.full(n, cid),
                           "split": np.full(n, split_of[row])})
            for key, v in values.items():
                if not np.all(np.isfinite(v)):
                    raise ValueError(f"cone {cid}: non-finite {key}")
                if d[key].dtype == np.float16 and np.abs(v).max() > 6.0e4:
                    raise ValueError(f"cone {cid}: {key} overflows float16")
                d[key].resize(n_rows + n, axis=0)
                d[key][n_rows:n_rows + n] = v.astype(d[key].dtype)
            n_rows += n
            log(f"[{k + 1}/{len(mine)}] cone {cid}: {n} slices, split {split_of[row]}")
        h.attrs.update({"schema": SCHEMA, "bands": json.dumps(bands), "param_names": json.dumps(param_names),
                        "cond_channels": json.dumps([f"density{b}" for b in bands] + [f"los_velocity{b}" for b in bands]),
                        "scalar_names": json.dumps(["1/(1+z)", *param_names] + [f"x_HI_global{b}" for b in bands]
                                                   + [f"T_b_global{b}" for b in bands]),
                        "structured": json.dumps(structured), "normalization": json.dumps(dataset.normalization),
                        "preparation": str(preparation), "history": history, "slices_per_cone": slices_per_cone,
                        "background_slices": background_slices, "xhi_window": json.dumps(list(xhi_window)),
                        "z_range": json.dumps(list(z_range)), "sample_seed": sample_seed,
                        "shard": shard, "n_shards": n_shards, "transverse_shape": json.dumps([H, W])})
    os.replace(tmp, out)
    log(f"wrote {n_rows} rows to {out}")


def merge(shards, out, log=print):
    shards = sorted(Path(s) for s in shards)
    fixed = ("schema", "bands", "param_names", "cond_channels", "scalar_names", "structured", "normalization",
             "preparation", "history", "slices_per_cone", "background_slices", "xhi_window", "z_range")
    with h5py.File(shards[0], "r") as h0:
        ref = {a: h0.attrs[a] for a in fixed}
        like = {k: (h0[k].shape[1:], h0[k].dtype, h0[k].chunks) for k in h0}
    total, seen = 0, set()
    for s in shards:
        with h5py.File(s, "r") as h:
            if any(h.attrs[a] != ref[a] for a in fixed):
                raise ValueError(f"{s}: attributes differ from {shards[0]}")
            ids = set(np.unique(h["cone_id"][:]).tolist())
            if ids & seen:
                raise ValueError(f"{s}: cone ids overlap another shard")
            seen |= ids
            total += h["z"].shape[0]
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    with h5py.File(tmp, "w") as o:
        for k, (shape, dt, chunks) in like.items():
            o.create_dataset(k, shape=(total, *shape), dtype=dt, chunks=chunks)
        pos = 0
        for s in shards:
            with h5py.File(s, "r") as h:
                n = h["z"].shape[0]
                for k in like:
                    for a in range(0, n, 256):
                        o[k][pos + a:pos + min(n, a + 256)] = h[k][a:min(n, a + 256)]
                pos += n
        o.attrs.update(ref)
        o.attrs.update({"merged_from": json.dumps([str(s) for s in shards]), "n_cones": len(seen)})
    os.replace(tmp, out)
    log(f"merged {len(shards)} shards, {total} rows from {len(seen)} cones -> {out}")


class MFMemorySplit:
    """One split of a multi-field slice cache in memory, batched and augmented on the device.

    ``batch`` returns ``cond`` (B, 26, H, W), ``scalars`` (B, 38), ``target``
    (B, 2, H, W: x_HI and normalized T_b) and ``phys`` (B, 1, H, W, mK), with
    one transverse dihedral element per sample applied to every field alike.
    """

    def __init__(self, path, split, storage="cpu", device="cpu", block_rows=256):
        self.path, self.split = str(path), split
        self.device, self.storage = torch.device(device), torch.device(storage)
        with h5py.File(self.path, "r") as h:
            if h.attrs.get("schema") != SCHEMA:
                raise ValueError(f"{path}: not an {SCHEMA} cache")
            rows = np.flatnonzero(h["split"][:] == SPLITS[split])
            self.attrs = dict(h.attrs)
            arrays = {k: np.empty((len(rows), *h[k].shape[1:]), np.float16) for k in ("cond", "phys", "xhi", "tb")}
            pos = 0
            for start, stop in _runs(rows):
                for a in range(start, stop, block_rows):
                    b = min(stop, a + block_rows)
                    for k, arr in arrays.items():
                        arr[pos:pos + b - a] = h[k][a:b]
                    pos += b - a
            self.scalars = torch.from_numpy(h["scalars"][:][rows].astype(np.float32)).to(self.storage)
            self.z, self.cone_id, self.rows = h["z"][:][rows], h["cone_id"][:][rows], rows
        self.cond, self.phys, self.xhi, self.tb = (torch.from_numpy(arrays[k]).to(self.storage)
                                                   for k in ("cond", "phys", "xhi", "tb"))
        self._z_t = torch.from_numpy(self.z.astype(np.float64))
        self._cone_t = torch.from_numpy(self.cone_id.astype(np.int64))

    def __len__(self):
        return len(self.rows)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.cond, self.phys, self.xhi, self.tb, self.scalars))

    def batch(self, positions, codes=None):
        positions = torch.as_tensor(positions, dtype=torch.int64)
        on = positions.to(self.storage)
        get = lambda t: t[on].to(self.device, non_blocking=True).float()
        cond = get(self.cond)
        target = torch.stack([get(self.xhi), get(self.tb)], 1)
        phys = get(self.phys)[:, None]
        if codes is not None:
            codes = torch.as_tensor(codes)
            for code in codes.unique().tolist():
                pick = (codes == code).nonzero()[:, 0].to(self.device)
                cond[pick], target[pick], phys[pick] = (dihedral(t[pick], code) for t in (cond, target, phys))
        return {"cond": cond, "scalars": self.scalars[on].to(self.device, non_blocking=True), "target": target,
                "phys": phys, "cone_id": self._cone_t[positions], "z": self._z_t[positions]}

    def loader(self, batch_size, positions=None):
        from .memory import _Loader

        return _Loader(self, batch_size, positions)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ebm21cm.data.mf_slices", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--shard", type=int, default=0)
    b.add_argument("--n-shards", type=int, default=1)
    b.add_argument("--preparation", default=PREPARATION)
    b.add_argument("--fno-root", default=FNO_ROOT)
    b.add_argument("--history", default=HISTORY)
    b.add_argument("--slices-per-cone", type=int, default=48)
    b.add_argument("--background-slices", type=int, default=6)
    b.add_argument("--limit", type=int, default=None)
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.add_argument("shards", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.out, a.shard, a.n_shards, a.preparation, a.fno_root, a.history, a.slices_per_cone,
              a.background_slices, limit=a.limit, log=lambda s: print(s, flush=True))
    else:
        merge(a.shards, a.out)


if __name__ == "__main__":
    sys.exit(main())
