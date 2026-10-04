"""Slice cache: one row per (cone, LOS index).

Each row holds the transverse target slice (x_HI, T_b) at native LOS index
``i`` and a conditioning stack of density *bands* around it. A band
``[lo, hi)`` is the mean of native density slices ``i+lo .. i+hi-1``, so
single slices near ``i`` and progressively wider LOS averages further out
can be mixed (bubbles are tens of Mpc across). Indices whose bands would
leave the cone are never sampled; no padding is invented.

Values are stored raw (physical units). Normalization lives in the
``/stats`` group, computed from training cones only, and is applied on the
fly by :class:`SliceDataset`. Splits are assigned per simulation, never per
slice.

Layout::

    delta      (N, C, H, W)   density band means
    xhi        (N, H, W)
    tb         (N, H, W)      mK
    z          (N,)           redshift of the target slice
    los_index  (N,)           native index, increasing-z order
    cone_id    (N,)
    params     (N, P)
    split      (N,)           0 train, 1 validation, 2 test
    attrs: bands, param_names, cell_size_mpc, shape, sources, split_seed, ...
    /stats     attrs: json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np

from .lightcone import DEFAULT_PARAM_NAMES, Lightcone

SCHEMA = "ebm21cm-slices-v1"
SPLITS = {"train": 0, "validation": 1, "val": 1, "test": 2}
# Single slices within +-3 cells, then LOS band means out to +-32 cells.
DEFAULT_BANDS = "-32:-16,-16:-8,-8:-4,-3,-2,-1,0,1,2,3,4:8,8:16,16:32"


# ---------------------------------------------------------------- bands
def parse_bands(spec: str) -> list[tuple[int, int]]:
    """``"-8:-4,0,4:8"`` -> ``[(-8, -4), (0, 1), (4, 8)]`` (half-open)."""
    bands = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            lo, hi = (int(t) for t in tok.split(":"))
        else:
            lo = int(tok)
            hi = lo + 1
        if hi <= lo:
            raise ValueError(f"empty band {tok!r}")
        bands.append((lo, hi))
    if not bands:
        raise ValueError("no bands")
    return bands


def band_extent(bands) -> tuple[int, int]:
    """(cells needed below i, cells needed above i)."""
    return max(0, -min(lo for lo, _ in bands)), max(0, max(hi for _, hi in bands) - 1)


def band_means(block: np.ndarray, center: int, bands) -> np.ndarray:
    """Band means from a (n, H, W) block where ``center`` indexes slice i."""
    return np.stack([block[center + lo:center + hi].mean(axis=0) for lo, hi in bands])


# ---------------------------------------------------------------- ids / splits
def cone_ids_for(files: list[Path]) -> list[int]:
    """Last digit run in each path (sample000123.h5, sim_00028/lightcone.h5)."""
    ids = []
    for p in files:
        runs = re.findall(r"\d+", str(Path(p.parent.name) / p.stem))
        ids.append(int(runs[-1]) if runs else -1)
    if len(set(ids)) != len(ids) or -1 in ids:
        print("warning: file names do not give unique ids; using sorted position", file=sys.stderr)
        ids = list(range(len(files)))
    return ids


def assign_splits(cone_ids, seed: int, fractions=(0.8, 0.1, 0.1)) -> dict[int, int]:
    ids = np.array(sorted(cone_ids))
    perm = np.random.default_rng(seed).permutation(len(ids))
    n = len(ids)
    n_val = max(1, int(round(fractions[1] * n))) if n >= 3 else 0
    n_test = max(1, int(round(fractions[2] * n))) if n >= 3 else 0
    split = {}
    for rank, j in enumerate(perm):
        split[int(ids[j])] = 1 if rank < n_val else 2 if rank < n_val + n_test else 0
    return split


def admissible_indices(z: np.ndarray, below: int, above: int, z_range) -> np.ndarray:
    idx = np.arange(below, len(z) - above)
    if z_range is not None:
        idx = idx[(z[idx] >= z_range[0]) & (z[idx] <= z_range[1])]
    return idx


def stratified(idx: np.ndarray, k: int, rng) -> np.ndarray:
    """k entries of ``idx``, one drawn uniformly from each of k equal index strata."""
    if idx.size <= k:
        return idx
    edges = np.linspace(0, idx.size, k + 1).astype(int)
    return np.array([idx[rng.integers(a, b)] for a, b in zip(edges[:-1], edges[1:]) if b > a])


def pick_indices(z: np.ndarray, below: int, above: int, k: int, z_range, rng) -> np.ndarray:
    """k LOS indices, stratified in index over the admissible range."""
    return stratified(admissible_indices(z, below, above, z_range), k, rng)


def stratified_levels(idx: np.ndarray, values: np.ndarray, k: int, rng) -> np.ndarray:
    """Up to k entries of ``idx`` whose ``values`` best match k stratified random levels.

    Levels are drawn one per equal-width stratum of [min, max] of ``values[idx]``,
    so the picks cover the value range evenly however slowly it changes along
    the LOS. Duplicate matches (steep jumps) are dropped rather than replaced.
    """
    if idx.size <= k:
        return idx
    v = values[idx]
    edges = np.linspace(v.min(), v.max(), k + 1)
    levels = rng.uniform(edges[:-1], edges[1:])
    return np.unique(idx[np.abs(v[None, :] - levels[:, None]).argmin(axis=1)])


def pick_window_indices(z, history, below, above, k, k_background, window, z_range, rng) -> np.ndarray:
    """k indices inside the cone's own reionization window, plus k_background outside it.

    ``history`` is the mean x_HI per LOS index. Inside
    ``window[0] <= history <= window[1]`` the rows are stratified in x_HI, not
    in LOS index: the history approaches 1 slowly at high z, and index strata
    would pile up early-phase slices. The remaining admissible indices are
    stratified in index. A cone that never enters the window contributes
    background slices only.
    """
    idx = admissible_indices(z, below, above, z_range)
    inside = (history[idx] >= window[0]) & (history[idx] <= window[1])
    picked = [stratified_levels(idx[inside], history, k, rng), stratified(idx[~inside], k_background, rng)]
    return np.sort(np.concatenate(picked)).astype(np.int64)


# ---------------------------------------------------------------- build
def discover(data: str | Path, pattern: str) -> list[Path]:
    root = Path(data)
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} under {root}")
    return files


def build(files, out, slices_per_cone=8, bands=DEFAULT_BANDS, z_range=None, split_seed=42,
          param_names=DEFAULT_PARAM_NAMES, missing_params="raise", dtype="float16",
          shard=0, n_shards=1, sample_seed=0, compute_stats_after=True, log=print,
          xhi_window=None, background_slices=0, chunk_cache_bytes=None, tb_overflow="skip",
          split_of=None, split_source=None):
    """``xhi_window=(lo, hi)`` concentrates ``slices_per_cone`` rows on the LOS
    indices where the cone's mean x_HI lies in [lo, hi] and adds
    ``background_slices`` rows from the rest of the admissible range. Without
    it, ``slices_per_cone`` rows are stratified over the whole range.

    ``tb_overflow="clip"`` keeps cones whose sampled T_b exceeds the float16
    range, clipping T_b to +-6e4 mK and listing them in the ``tb_clipped``
    attribute (their T_b is then excluded from the /stats normalization).
    The default ``"skip"`` drops such cones. x_HI and density are never clipped.

    ``split_of`` ({cone id: 0/1/2}, e.g. from ``split_from_preparation``) replaces
    the seeded 80/10/10 split; cones it does not list are not read.
    """
    if tb_overflow not in ("skip", "clip"):
        raise ValueError("tb_overflow must be 'skip' or 'clip'")
    files = [Path(f) for f in files]
    if xhi_window is not None:
        xhi_window = tuple(float(v) for v in xhi_window)
        if len(xhi_window) != 2 or not 0 <= xhi_window[0] <= xhi_window[1] <= 1:
            raise ValueError("xhi_window must be (lo, hi) with 0 <= lo <= hi <= 1")
    bands_l = parse_bands(bands) if isinstance(bands, str) else [tuple(b) for b in bands]
    below, above = band_extent(bands_l)
    ids = cone_ids_for(files)
    if split_of is None:
        split_of = assign_splits(ids, split_seed)  # global, before sharding
    else:
        split_of = {int(k): int(v) for k, v in split_of.items()}
        missing = set(split_of) - set(ids)
        if missing:
            raise ValueError(f"{len(missing)} cones of the given split have no file, e.g. {sorted(missing)[:5]}")
        files, ids = map(list, zip(*[(f, c) for f, c in zip(files, ids) if c in split_of]))
    mine = [(f, c) for j, (f, c) in enumerate(zip(files, ids)) if j % n_shards == shard]

    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    store_dtype = np.dtype(dtype)
    n_rows = 0
    geometry = None
    skipped, tb_clipped = [], []
    with h5py.File(tmp, "w") as h:
        dsets = {}
        for f, cid in mine:
            try:
                with Lightcone(f, chunk_cache_bytes) as lc:
                    H, W = lc.shape_hw
                    geo = (H, W, round(lc.cell_size, 6))
                    if geometry is None:
                        geometry = geo
                        C = len(bands_l)
                        mk = lambda name, shape, dt: h.create_dataset(
                            name, shape=(0, *shape), maxshape=(None, *shape), dtype=dt,
                            chunks=(1, *shape) if shape else (1024,))
                        dsets = {
                            "delta": mk("delta", (C, H, W), store_dtype),
                            "xhi": mk("xhi", (H, W), store_dtype),
                            "tb": mk("tb", (H, W), store_dtype),
                            "z": mk("z", (), np.float64),
                            "los_index": mk("los_index", (), np.int32),
                            "los_spacing_mpc": mk("los_spacing_mpc", (), np.float64),
                            "cone_id": mk("cone_id", (), np.int64),
                            "split": mk("split", (), np.int8),
                        }
                        dsets["params"] = h.create_dataset(
                            "params", shape=(0, len(param_names)), maxshape=(None, len(param_names)),
                            dtype=np.float64, chunks=(1024, len(param_names)))
                    elif geo != geometry:
                        skipped.append((str(f), f"geometry {geo} != {geometry}"))
                        continue
                    params = lc.params(param_names, missing=missing_params)
                    rng = np.random.default_rng([sample_seed, cid])
                    if xhi_window is None:
                        idx = pick_indices(lc.redshifts, below, above, slices_per_cone, z_range, rng)
                    else:
                        idx = pick_window_indices(lc.redshifts, lc.xhi_history(), below, above,
                                                  slices_per_cone, background_slices, xhi_window,
                                                  z_range, rng)
                    if idx.size == 0:
                        skipped.append((str(f), "no admissible LOS index"))
                        continue
                    rows = {k: [] for k in ("delta", "xhi", "tb")}
                    for i in idx:
                        block = lc.read_range("density", i - below, i + above + 1)
                        rows["delta"].append(band_means(block, below, bands_l))
                        rows["xhi"].append(lc.read_range("neutral_fraction", i, i + 1)[0])
                        rows["tb"].append(lc.read_range("brightness_temp", i, i + 1)[0])
                    clipped = False
                    for k, v in list(rows.items()):
                        v = np.stack(v)
                        if not np.all(np.isfinite(v)):
                            raise ValueError(f"non-finite {k}")
                        if store_dtype == np.float16 and np.abs(v).max() > 6.0e4:
                            if k != "tb" or tb_overflow != "clip":
                                raise ValueError(f"{k} overflows float16")
                            rows[k], clipped = np.clip(v, -6.0e4, 6.0e4), True
                    dz = np.gradient(lc.distances)
                    n = idx.size
                    extra = {
                        "z": lc.redshifts[idx], "los_index": idx,
                        "los_spacing_mpc": dz[idx], "cone_id": np.full(n, cid),
                        "split": np.full(n, split_of[cid]), "params": np.tile(params, (n, 1)),
                    }
            except (OSError, KeyError, ValueError, IndexError) as e:
                skipped.append((str(f), f"{type(e).__name__}: {e}"))
                continue
            for k, v in list(rows.items()) + list(extra.items()):
                ds = dsets[k]
                ds.resize(n_rows + n, axis=0)
                ds[n_rows:n_rows + n] = np.asarray(v).astype(ds.dtype)
            n_rows += n
            if clipped:
                tb_clipped.append(int(cid))
            log(f"cone {cid}: {n} slices, z {extra['z'].min():.2f}-{extra['z'].max():.2f}, "
                f"split {split_of[cid]}")
        if geometry is None:
            raise RuntimeError(f"no usable cones; skipped: {skipped}")
        h.attrs.update({
            "schema": SCHEMA, "bands": json.dumps(bands_l), "param_names": json.dumps(list(param_names)),
            "cell_size_mpc": geometry[2], "shape": json.dumps(geometry[:2]),
            "split_seed": split_seed, "sample_seed": sample_seed, "slices_per_cone": slices_per_cone,
            "z_range": json.dumps(z_range), "shard": shard, "n_shards": n_shards,
            "xhi_window": json.dumps(xhi_window), "background_slices": int(background_slices),
            "sources": json.dumps([str(f) for f, _ in mine]), "skipped": json.dumps(skipped),
            "tb_clipped": json.dumps(tb_clipped),
            "split_of": json.dumps({str(k): v for k, v in split_of.items()}),
            "split_source": json.dumps(split_source),
        })
    os.replace(tmp, out)
    for f, why in skipped:
        log(f"skipped {f}: {why}")
    if tb_clipped:
        log(f"T_b clipped to float16 range in cones {tb_clipped}")
    log(f"wrote {n_rows} rows to {out}")
    if compute_stats_after and n_shards == 1:
        write_stats(out, log=log)
    return out


def split_from_preparation(path) -> dict[int, int]:
    """{cone id: 0 train / 1 val / 2 test} of an fno-21cm LOS-window preparation JSON."""
    prep = json.loads(Path(path).read_text())
    ids = prep["source"]["cone_ids"]
    codes = {"train": 0, "val": 1, "test": 2}
    return {int(ids[row]): codes[name] for name, rows in prep["split"].items() for row in rows}


def merge(shards, out, log=print):
    shards = [Path(s) for s in shards]
    keys = ("delta", "xhi", "tb", "z", "los_index", "los_spacing_mpc", "cone_id", "split", "params")
    fixed = ("schema", "bands", "param_names", "cell_size_mpc", "shape", "split_seed",
             "sample_seed", "slices_per_cone", "z_range", "split_of", "xhi_window", "background_slices",
             "split_source")
    with h5py.File(shards[0], "r") as h0:
        # Caches written before the reionization-window option lack its attributes.
        ref = {a: h0.attrs[a] for a in fixed if a in h0.attrs}
        like = {k: (h0[k].shape[1:], h0[k].dtype, h0[k].chunks) for k in keys}
    seen, total = set(), 0
    for s in shards:
        with h5py.File(s, "r") as h:
            for a in fixed:
                if a in ref and h.attrs.get(a) != ref[a]:
                    raise ValueError(f"{s}: attribute {a} differs from {shards[0]}")
            ids = set(np.unique(h["cone_id"][:]).tolist())
            if ids & seen:
                raise ValueError(f"{s}: cone ids overlap another shard")
            seen |= ids
            total += h["z"].shape[0]
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    with h5py.File(tmp, "w") as o:
        for k, (shape, dt, chunks) in like.items():
            chunks = (min(chunks[0], max(total, 1)), *chunks[1:]) if chunks else None
            o.create_dataset(k, shape=(total, *shape), dtype=dt, chunks=chunks)
        pos, sources, skipped, tb_clipped = 0, [], [], []
        for s in shards:
            with h5py.File(s, "r") as h:
                n = h["z"].shape[0]
                for k in keys:
                    for a in range(0, n, 256):
                        b = min(n, a + 256)
                        o[k][pos + a:pos + b] = h[k][a:b]
                sources += json.loads(h.attrs["sources"])
                skipped += json.loads(h.attrs["skipped"])
                tb_clipped += json.loads(h.attrs.get("tb_clipped", "[]"))
                pos += n
        o.attrs.update(ref)
        o.attrs.update({"sources": json.dumps(sources), "skipped": json.dumps(skipped),
                        "tb_clipped": json.dumps(tb_clipped),
                        "shard": 0, "n_shards": 1, "merged_from": json.dumps([str(s) for s in shards])})
    os.replace(tmp, out)
    log(f"merged {len(shards)} shards, {total} rows -> {out}")
    write_stats(out, log=log)
    return out


# ---------------------------------------------------------------- stats
def delta_transform(d):
    """log(1 + delta), guarded against the -1 floor."""
    return np.log1p(np.maximum(d, -0.99))


def write_stats(path, chunk=256, log=print):
    """Normalization from training rows only; stored as JSON in /stats."""
    with h5py.File(path, "r+") as h:
        train = np.flatnonzero(h["split"][:] == 0)
        if train.size == 0:
            raise ValueError("no training rows")
        # Clipped T_b values are not physical; keep them out of its normalization.
        clipped = np.isin(h["cone_id"][:][train], json.loads(h.attrs.get("tb_clipped", "[]")))
        acc = {k: [0.0, 0.0, 0] for k in ("delta", "tb")}
        xhi_acc = [0.0, 0.0, 0]
        for a in range(0, train.size, chunk):
            rows = train[a:a + chunk]
            for key, fn in (("delta", delta_transform), ("tb", lambda v: v)):
                v = fn(h[key][rows].astype(np.float64))
                if key == "tb":
                    v = v[~clipped[a:a + chunk]]
                acc[key][0] += v.sum()
                acc[key][1] += (v ** 2).sum()
                acc[key][2] += v.size
            x = 2.0 * h["xhi"][rows].astype(np.float64) - 1.0
            xhi_acc[0] += x.sum()
            xhi_acc[1] += (x ** 2).sum()
            xhi_acc[2] += x.size

        def ms(a):
            m = a[0] / a[2]
            return m, float(np.sqrt(max(a[1] / a[2] - m * m, 1e-24)))

        d_mu, d_sd = ms(acc["delta"])
        tb_mu, tb_sd = ms(acc["tb"])
        # After the target transforms below, both channels are ~unit scale;
        # sigma_data is their joint RMS, as EDM preconditioning expects.
        x_mu, x_sd = ms(xhi_acc)
        sigma_data = float(np.sqrt(0.5 * ((x_sd ** 2 + x_mu ** 2) + 1.0)))
        lz = np.log1p(h["z"][train])
        params = h["params"][train]
        stats = {
            "delta_log1p_mean": d_mu, "delta_log1p_std": d_sd,
            "tb_mean": tb_mu, "tb_std": tb_sd,
            "log1pz_mean": float(lz.mean()), "log1pz_std": float(max(lz.std(), 1e-6)),
            "params_min": np.nanmin(params, axis=0).tolist(),
            "params_max": np.nanmax(params, axis=0).tolist(),
            "sigma_data": sigma_data, "n_train_rows": int(train.size),
            "n_train_cones": int(np.unique(h["cone_id"][train]).size),
        }
        if "stats" in h:
            del h["stats"]
        g = h.create_group("stats")
        g.attrs["json"] = json.dumps(stats)
    log(f"stats: {json.dumps(stats)}")
    return stats


def read_stats(path) -> dict:
    with h5py.File(path, "r") as h:
        if "stats" not in h:
            raise KeyError(f"{path} has no /stats; run `python -m ebm21cm.data.cache stats`")
        return json.loads(h["stats"].attrs["json"])


# ---------------------------------------------------------------- dataset
class Normalizer:
    """Physical <-> network coordinates. Pure numpy/torch arithmetic."""

    def __init__(self, stats: dict):
        self.s = stats
        lo = np.asarray(stats["params_min"], dtype=np.float64)
        hi = np.asarray(stats["params_max"], dtype=np.float64)
        span = np.where(hi > lo, hi - lo, 1.0)
        self.p_lo, self.p_span = lo, span

    def cond(self, delta):
        return (delta_transform(delta) - self.s["delta_log1p_mean"]) / self.s["delta_log1p_std"]

    def target(self, xhi, tb):
        return np.stack([2.0 * xhi - 1.0, (tb - self.s["tb_mean"]) / self.s["tb_std"]])

    def scalars(self, z, params):
        """(log(1+z), params) -> standardized z and params in [-1, 1]; batched or not."""
        z = np.asarray(z, dtype=np.float64)
        lz = (np.log1p(z) - self.s["log1pz_mean"]) / self.s["log1pz_std"]
        p = 2.0 * (np.asarray(params, dtype=np.float64) - self.p_lo) / self.p_span - 1.0
        return np.concatenate([lz[..., None], np.nan_to_num(p, nan=0.0)], axis=-1)

    def to_physical(self, y):
        """Network targets (..., 2, H, W) -> (x_HI in [0,1], T_b in mK)."""
        xhi = (y[..., 0, :, :] + 1.0) / 2.0
        tb = y[..., 1, :, :] * self.s["tb_std"] + self.s["tb_mean"]
        return xhi.clip(0.0, 1.0), tb


def augment(cond, target, rng):
    """Random transverse dihedral transform + periodic roll, applied jointly.

    21cmFAST transverse planes come from periodic boxes and the statistics are
    isotropic across the sky plane, so both are exact symmetries of the data.
    """
    k = int(rng.integers(4))
    flip = bool(rng.integers(2))
    sx, sy = (int(v) for v in rng.integers(0, cond.shape[-1], size=2))
    out = []
    for a in (cond, target):
        a = np.rot90(a, k, axes=(-2, -1))
        if flip:
            a = a[..., ::-1]
        a = np.roll(a, (sx, sy), axis=(-2, -1))
        out.append(np.ascontiguousarray(a))
    return out


class SliceDataset:
    """torch-style dataset over one split of a slice cache."""

    def __init__(self, path, split="train", stats=None, augment_data=False, seed=0, rows=None):
        self.path = str(path)
        with h5py.File(self.path, "r") as h:
            if h.attrs.get("schema") != SCHEMA:
                raise ValueError(f"{path}: not an {SCHEMA} cache")
            code = SPLITS[split]
            self.rows = np.flatnonzero(h["split"][:] == code) if rows is None else np.asarray(rows)
            self.attrs = {k: h.attrs[k] for k in ("bands", "param_names", "cell_size_mpc", "shape")}
            self.z = h["z"][:][self.rows]
            self.cone_id = h["cone_id"][:][self.rows]
        self.stats = stats if stats is not None else read_stats(self.path)
        self.norm = Normalizer(self.stats)
        self.augment = augment_data
        self.seed = seed
        self._h = None
        self._rng = None

    def __len__(self):
        return len(self.rows)

    @property
    def n_cond(self):
        return len(json.loads(self.attrs["bands"]))

    @property
    def center_band(self):
        """Index of the band containing the target slice itself."""
        bands = json.loads(self.attrs["bands"])
        hits = [j for j, (lo, hi) in enumerate(bands) if lo <= 0 < hi]
        return hits[0] if hits else len(bands) // 2

    @property
    def n_scalars(self):
        return 1 + len(json.loads(self.attrs["param_names"]))

    def raw(self, j):
        if self._h is None:  # opened lazily so each DataLoader worker has its own handle
            self._h = h5py.File(self.path, "r")
        r = int(self.rows[j])
        h = self._h
        return {k: h[k][r] for k in ("delta", "xhi", "tb", "z", "params", "cone_id", "los_index")}

    def __getitem__(self, j):
        import torch

        d = self.raw(j)
        cond = self.norm.cond(d["delta"].astype(np.float64)).astype(np.float32)
        target = self.norm.target(d["xhi"].astype(np.float64), d["tb"].astype(np.float64)).astype(np.float32)
        if self.augment:
            if self._rng is None:
                info = None
                try:
                    info = torch.utils.data.get_worker_info()
                except Exception:
                    pass
                self._rng = np.random.default_rng([self.seed, info.id if info else 0])
            cond, target = augment(cond, target, self._rng)
        scal = self.norm.scalars(float(d["z"]), d["params"]).astype(np.float32)
        return {
            "cond": torch.from_numpy(cond), "target": torch.from_numpy(target),
            "scalars": torch.from_numpy(scal), "z": float(d["z"]),
            "cone_id": int(d["cone_id"]), "row": int(self.rows[j]),
        }

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_h"] = None
        st["_rng"] = None
        return st


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ebm21cm.data.cache", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build a slice cache (or one shard) from lightcones")
    b.add_argument("--data", required=True, help="directory holding lightcone .h5 files")
    b.add_argument("--pattern", default="*.h5", help="glob under --data (e.g. 'sim_*/lightcone.h5')")
    b.add_argument("--out", required=True)
    b.add_argument("--slices-per-cone", type=int, default=8)
    b.add_argument("--bands", default=DEFAULT_BANDS)
    b.add_argument("--z-min", type=float)
    b.add_argument("--z-max", type=float)
    b.add_argument("--split-seed", type=int, default=42)
    b.add_argument("--split-from", help="fno-21cm preparation JSON whose train/val/test cones to use "
                   "(replaces the seeded split; other cones are skipped)")
    b.add_argument("--sample-seed", type=int, default=0)
    b.add_argument("--param-names", default=",".join(DEFAULT_PARAM_NAMES))
    b.add_argument("--missing-params", choices=("raise", "nan"), default="raise")
    b.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    b.add_argument("--shard", type=int, default=0)
    b.add_argument("--n-shards", type=int, default=1)
    b.add_argument("--limit", type=int, help="use only the first N files (pilot runs)")
    b.add_argument("--xhi-window", help="'lo,hi': sample --slices-per-cone rows where the cone's "
                   "mean x_HI lies in [lo, hi] (e.g. 0.02,0.98)")
    b.add_argument("--background-slices", type=int, default=0,
                   help="with --xhi-window: extra rows stratified over the rest of the cone")
    b.add_argument("--tb-overflow", choices=("skip", "clip"), default="skip",
                   help="cones whose T_b exceeds float16: drop them, or clip T_b and keep them")
    b.add_argument("--chunk-cache-mb", type=int, default=1024,
                   help="HDF5 chunk cache per dataset while reading lightcones")
    m = sub.add_parser("merge", help="merge shards and compute stats")
    m.add_argument("--out", required=True)
    m.add_argument("shards", nargs="+")
    s = sub.add_parser("stats", help="(re)compute training statistics")
    s.add_argument("cache")
    a = ap.parse_args(argv)

    if a.cmd == "build":
        files = discover(a.data, a.pattern)
        if a.limit:
            files = files[:a.limit]
        zr = None if a.z_min is None and a.z_max is None else (
            a.z_min if a.z_min is not None else -np.inf, a.z_max if a.z_max is not None else np.inf)
        window = tuple(float(v) for v in a.xhi_window.split(",")) if a.xhi_window else None
        build(files, a.out, a.slices_per_cone, a.bands, zr, a.split_seed,
              tuple(n for n in a.param_names.split(",") if n), a.missing_params, a.dtype,
              a.shard, a.n_shards, a.sample_seed, xhi_window=window,
              background_slices=a.background_slices, chunk_cache_bytes=a.chunk_cache_mb * 2**20,
              tb_overflow=a.tb_overflow,
              split_of=split_from_preparation(a.split_from) if a.split_from else None,
              split_source=a.split_from)
    elif a.cmd == "merge":
        merge(a.shards, a.out)
    else:
        write_stats(a.cache)


if __name__ == "__main__":
    main()
