"""LoReLi II slice cache for fine-tuning.

The cache is built with the *evaluation* adapter, so a training row is exactly
the tensor the reference model receives when the same cone and target slice are
evaluated: 26 normalized density/velocity bands, 38 scalars, and the structured
head's physical factor on the 280-cell domain.

Targets are resampled to the same 206-cell (L=294) / 1.4286 Mpc grid, then
mirror-padded to 280 exactly like the conditioning.  A single 2-D mask marks the
206-cell interior; the mirrored border is stored but ignored by the loss and by
every metric.

Row schema (per slice)::

    cond      (26, 280, 280) float16
    target    (2, 280, 280)  float16   x_HI (continuous) and normalized T_b
    phys      (280, 280)     float32   A(z)(1+delta)V with V=1, mK
    scalars   (38,)          float32   1/(1+z), 11 training-mean params, 13+13 histories
    mask      (280, 280)     uint8     shared; 1 inside the valid 206x206 interior
    z, los_index, cone_id, block, split, xhi_mean, stage, background

The 11 21cmFAST parameters are zero in normalized space (the training mean);
the histories are LoReLi's own.  ``parameter_mode`` records that choice.

Build::

    python -m ebm21cm.data.loreli_cache build \
        --out runs/loreli_cache/loreli_slices_l294.h5 \
        --data /pfs/.../data/loreli_lc128 \
        --training-cache /pfs/.../compressed/rhizome_mf_slices_all.h5 \
        --run /pfs/.../runs/rhizome_mf2d_w64_structured_all \
        --split-file runs/loreli_splits/block_split_seed0.json \
        --l-perp 294 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .loreli import (
    TRAINING_CELL_MPC,
    LoreliAdapter,
    LoreliCone,
    loreli_files,
    pad_to,
    read_cube,
    resample_los,
    resample_transverse,
    training_stats,
)
from .loreli_splits import block_of, load_split
from ..loreli_metrics import stage_of
from ..run import git_commit

SCHEMA = "ebm21cm-loreli-slices-v1"
SPLIT_ID = {"train": 0, "val": 1, "test": 2}
DOMAIN = 280
ACTIVE_RANGE = (0.02, 0.98)


def padding_mask(domain: int = DOMAIN, n_out: int | None = None, l_perp: float = 294.0) -> np.ndarray:
    """Centered transverse mask: True on the valid ``n_out x n_out`` interior."""
    if n_out is None:
        n_out = int(round(float(l_perp) / TRAINING_CELL_MPC))
    if not 0 < n_out <= domain:
        raise ValueError(f"n_out={n_out} does not fit domain={domain}")
    pad = (domain - n_out) // 2
    mask = np.zeros((domain, domain), np.uint8)
    mask[pad:pad + n_out, pad:pad + n_out] = 1
    return mask


def select_target_indices(
    xhi_mean: np.ndarray,
    active: tuple[float, float] = ACTIVE_RANGE,
    background: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """All active target slices plus a deterministic background sample.

    Returns ``(indices, is_background)`` sorted by index.  A background slice is
    outside ``active`` (mostly ionized or mostly neutral); the sample is evenly
    spaced over the available background indices, like the 21cmFAST cache.
    """
    mean = np.asarray(xhi_mean, dtype=np.float64)
    if mean.ndim != 1 or mean.size == 0:
        raise ValueError("xhi_mean must be a non-empty 1-D array")
    in_active = (mean >= active[0]) & (mean <= active[1])
    active_idx = np.flatnonzero(in_active)
    bg_idx = np.flatnonzero(~in_active)
    if active_idx.size == 0:
        raise ValueError("no active slices (0.02 <= xHI <= 0.98) in this cone")
    chosen = list(active_idx)
    if background > 0 and bg_idx.size:
        picks = np.unique(
            np.rint(np.linspace(0, bg_idx.size - 1, min(int(background), bg_idx.size))).astype(int)
        )
        chosen += [bg_idx[i] for i in picks]
    order = np.argsort(chosen)
    indices = np.asarray(chosen, dtype=np.int64)[order]
    is_background = ~in_active[indices]
    return indices, is_background


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contiguous_groups(indices: np.ndarray):
    indices = np.asarray(indices, dtype=np.int64)
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    return np.split(indices, breaks)


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _resampled_truth(cone: LoreliCone, adapter: LoreliAdapter):
    """Truth x_HI and T_b on the adapter's 206-cell / 1.4286 Mpc target grid."""
    raw = cone.read(0, cone.n_los)
    truth_x = 1.0 - np.asarray(raw["xion"], np.float32)
    truth_tb = np.asarray(raw["dtb"], np.float32)
    x_los = resample_los(truth_x, cone.distances, adapter.target_distance)
    tb_los = resample_los(truth_tb, cone.distances, adapter.target_distance)
    return (
        resample_transverse(x_los, adapter.n_out),
        resample_transverse(tb_los, adapter.n_out),
    )


def rows_from_cone(
    cone: LoreliCone,
    adapter: LoreliAdapter,
    tb_norm: tuple[float, float],
    *,
    domain: int = DOMAIN,
    active: tuple[float, float] = ACTIVE_RANGE,
    background: int = 2,
    chunk: int = 8,
    indices: np.ndarray | None = None,
) -> tuple[dict, dict]:
    """Build every cache row for one cone; returns ``(arrays, metadata)``.

    ``arrays`` has the per-row columns (not yet cast to storage dtypes).  The
    conditioning/scalars/physics come from :meth:`LoreliAdapter.inputs`, so they
    are identical to the evaluation path for the same target index.
    """
    x_target, tb_target = _resampled_truth(cone, adapter)
    mean = x_target.mean(axis=(0, 1))
    if indices is None:
        indices, is_background = select_target_indices(mean, active, background)
    else:
        indices = np.sort(np.asarray(indices, dtype=np.int64))
        is_background = ~((mean[indices] >= active[0]) & (mean[indices] <= active[1]))
    if indices.size == 0:
        raise ValueError("empty target selection")
    if indices.min() < 0 or indices.max() >= len(adapter.target_distance):
        raise ValueError("target index outside the adapter's target grid")

    offset, scale = (float(v) for v in tb_norm)
    target_x = np.clip(x_target[..., indices], 0.0, 1.0)
    target_tb = (tb_target[..., indices] - offset) / scale
    expected_channels = 2 * len(adapter.bands)

    conds, targets, physs, scalarss, zs, los, means, stages, flags = (
        [], [], [], [], [], [], [], [], []
    )
    selected_position = {int(index): i for i, index in enumerate(indices)}
    chunk = max(1, int(chunk))
    for group in _contiguous_groups(indices):
        for start in range(int(group[0]), int(group[-1]) + 1, chunk):
            stop = min(int(group[-1]) + 1, start + chunk)
            chosen = group[(group >= start) & (group < stop)]
            if chosen.size == 0:
                continue
            take = chosen - start
            inp = adapter.inputs(start, stop)
            cond = _numpy(inp["cond"])
            scalars = _numpy(inp["scalars"])
            phys = _numpy(inp["phys"])
            z = _numpy(inp["z"])
            if cond.ndim != 4 or cond.shape[1] != expected_channels:
                raise ValueError(
                    f"adapter cond must be (n,{expected_channels},H,W), got {cond.shape}"
                )
            if cond.shape[-2:] != (domain, domain):
                raise ValueError(f"adapter domain {cond.shape[-2:]} != cache domain {domain}")
            cols = np.array([selected_position[int(index)] for index in chosen])
            x = target_x[:, :, cols]
            t = target_tb[:, :, cols]
            x = np.moveaxis(pad_to(x, domain), -1, 0)[:, None]
            t = np.moveaxis(pad_to(t, domain), -1, 0)[:, None]
            target = np.concatenate([x, t], axis=1)
            conds.append(cond[take].astype(np.float16))
            targets.append(target.astype(np.float16))
            physs.append(phys[take, 0].astype(np.float32))
            scalarss.append(scalars[take].astype(np.float32))
            zs.append(z[take].astype(np.float64))
            los.append(chosen.astype(np.int32))
            means.append(mean[chosen].astype(np.float32))
            stages.append(np.array(
                [-1 if stage_of(v) is None else {"xHI_0.05-0.35": 0, "xHI_0.35-0.65": 1,
                                                 "xHI_0.65-0.95": 2}[stage_of(v)]
                 for v in mean[chosen]], np.int8,
            ))
            flags.append(is_background[cols])

    arrays = {
        "cond": np.concatenate(conds, 0),
        "target": np.concatenate(targets, 0),
        "phys": np.concatenate(physs, 0),
        "scalars": np.concatenate(scalarss, 0),
        "z": np.concatenate(zs, 0),
        "los_index": np.concatenate(los, 0),
        "xhi_mean": np.concatenate(means, 0),
        "stage": np.concatenate(stages, 0),
        "background": np.concatenate(flags, 0),
    }
    meta = {
        "cone_id": int(cone.sim),
        "n_rows": int(indices.size),
        "n_active": int(np.count_nonzero(~np.concatenate(flags, 0))),
    }
    return arrays, meta


def build(
    out: str | Path,
    data: str | Path,
    training_cache: str | Path,
    run: str | Path,
    split_file: str | Path,
    *,
    l_perp: float = 294.0,
    domain: int = DOMAIN,
    active: tuple[float, float] = ACTIVE_RANGE,
    background: int = 2,
    device: str = "cuda",
    chunk: int = 8,
    limit: int | None = None,
    sims=None,
    parameter_mode: str = "training-mean",
    overwrite: bool = False,
    log=print,
) -> dict:
    """Build the HDF5 cache.  Returns a small provenance dictionary."""
    import torch

    if parameter_mode != "training-mean":
        raise ValueError("only parameter_mode='training-mean' is supported")
    out = Path(out)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)

    stats = training_stats(training_cache)
    if 2 * len(stats["bands"]) != 26:
        raise ValueError(f"{training_cache}: expected 13 bands, got {len(stats['bands'])}")
    if len(stats["scalar_names"]) != 38:
        raise ValueError(f"{training_cache}: expected 38 scalars, got {len(stats['scalar_names'])}")
    split = load_split(split_file)
    best = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
    if best.get("head") != "structured":
        raise ValueError(f"{run}: fine-tuning expects the structured T_b head")
    tb_norm = tuple(float(v) for v in best["tb_norm"])
    base_config = dict(best["model_config"])

    files, file_sims = loreli_files(data)
    keep = []
    unassigned = []
    for path, sim in zip(files, file_sims):
        if sims is not None and int(sim) not in set(int(s) for s in sims):
            continue
        if int(sim) not in split["sim_to_split"]:
            unassigned.append(int(sim))
        keep.append((path, int(sim)))
    if limit:
        keep = keep[: int(limit)]
    if not keep:
        raise SystemExit("no cones selected")
    log(f"building {out}: {len(keep)} cones ({len(unassigned)} unassigned, stored split=-1), "
        f"{split['train_blocks'].__len__()} train / {split['val_blocks'].__len__()} val / "
        f"{split['test_blocks'].__len__()} test blocks; l_perp={l_perp}, domain={domain}, device={device}")

    n_out = int(round(float(l_perp) / TRAINING_CELL_MPC))
    mask = padding_mask(domain, n_out)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    skipped = []
    n_rows = 0
    started = time.time()
    with h5py.File(tmp, "w") as h:
        def mk(name, shape, dtype, chunks=None):
            return h.create_dataset(
                name, shape=(0, *shape), maxshape=(None, *shape), dtype=dtype,
                chunks=(1, *shape) if len(shape) > 1 else chunks or (1024,),
            )

        d = {
            "cond": mk("cond", (26, domain, domain), np.float16),
            "target": mk("target", (2, domain, domain), np.float16),
            "phys": mk("phys", (domain, domain), np.float32),
            "scalars": mk("scalars", (38,), np.float32, chunks=(1024, 38)),
            "z": mk("z", (), np.float64),
            "los_index": mk("los_index", (), np.int32),
            "cone_id": mk("cone_id", (), np.int64),
            "block": mk("block", (), np.int32),
            "split": mk("split", (), np.int8),
            "xhi_mean": mk("xhi_mean", (), np.float32),
            "stage": mk("stage", (), np.int8),
            "background": mk("background", (), np.bool_),
        }
        h.create_dataset("mask", data=mask)

        for k, (path, sim) in enumerate(keep):
            t0 = time.time()
            try:
                cone = LoreliCone(
                    path, l_perp=l_perp, density_mean="slice", n_native=128,
                    n_los=1536, data=read_cube(path),
                )
                adapter = LoreliAdapter(
                    cone, stats, velocity="linear", l_perp=l_perp,
                    device=device, force_torch=device != "cpu",
                )
                arrays, meta = rows_from_cone(
                    cone, adapter, tb_norm, domain=domain, active=active,
                    background=background, chunk=chunk,
                )
                n = arrays["cond"].shape[0]
                if n == 0:
                    raise ValueError("no rows")
                if not np.all(np.isfinite(arrays["cond"])) or not np.all(np.isfinite(arrays["target"])):
                    raise ValueError("non-finite cache tensor")
                if np.abs(arrays["cond"]).max() > 6.0e4 or np.abs(arrays["target"]).max() > 6.0e4:
                    raise ValueError("float16 overflow")
                block = block_of(sim)
                split_name = split["block_to_split"].get(block)
                split_id = -1 if split_name is None else SPLIT_ID[split_name]
                values = {
                    "cond": arrays["cond"],
                    "target": arrays["target"],
                    "phys": arrays["phys"],
                    "scalars": arrays["scalars"],
                    "z": arrays["z"],
                    "los_index": arrays["los_index"],
                    "cone_id": np.full(n, sim, np.int64),
                    "block": np.full(n, block, np.int32),
                    "split": np.full(n, split_id, np.int8),
                    "xhi_mean": arrays["xhi_mean"],
                    "stage": arrays["stage"],
                    "background": arrays["background"],
                }
                for key, value in values.items():
                    d[key].resize(n_rows + n, axis=0)
                    d[key][n_rows:n_rows + n] = value.astype(d[key].dtype)
                n_rows += n
                log(f"[{k + 1}/{len(keep)}] sim {sim} block {block} "
                    f"{split_name or 'unassigned'}: {n} rows "
                    f"({meta['n_active']} active), {time.time() - t0:.1f} s")
            except (ValueError, OSError, KeyError) as exc:
                skipped.append([int(sim), f"{type(exc).__name__}: {exc}"])
                log(f"[{k + 1}/{len(keep)}] sim {sim}: skipped ({exc})")

        h.attrs.update(
            {
                "schema": SCHEMA,
                "created": datetime.now(timezone.utc).isoformat(),
                "data": str(Path(data).resolve()),
                "training_cache": str(Path(training_cache).resolve()),
                "reference_run": str(Path(run).resolve()),
                "checkpoint_sha256": _sha256(Path(run) / "best.pt"),
                "checkpoint_step": int(best.get("step", -1)),
                "base_model_config": json.dumps(base_config),
                "tb_norm": json.dumps(tb_norm),
                "head": str(best.get("head")),
                "bands": json.dumps(stats["bands"]),
                "normalization": json.dumps(stats["normalization"]),
                "param_names": json.dumps(stats["param_names"]),
                "scalar_names": json.dumps(stats["scalar_names"]),
                "l_perp": float(l_perp),
                "domain": int(domain),
                "n_out": int(n_out),
                "pad": int((domain - n_out) // 2),
                "active_range": json.dumps(list(active)),
                "background_slices": int(background),
                "parameter_mode": parameter_mode,
                "split_file": str(Path(split_file).resolve()),
                "split_sha256": split["sha256"],
                "n_cones": len(keep) - len(skipped),
                "n_unassigned": len(unassigned),
                "unassigned_sims": json.dumps(sorted(unassigned)),
                "n_rows": int(n_rows),
                "skipped": json.dumps(skipped),
                "git_commit": git_commit(Path(__file__).resolve().parents[2]),
            }
        )
    os.replace(tmp, out)
    elapsed = time.time() - started
    size = out.stat().st_size
    log(f"wrote {n_rows} rows, {size / 1e9:.1f} GB in {elapsed / 3600:.2f} h; {len(skipped)} cones skipped")
    return {"path": str(out), "n_rows": n_rows, "size_bytes": size, "seconds": elapsed, "skipped": skipped}


def _parse_pair(values):
    if values is None:
        return ACTIVE_RANGE
    return (float(values[0]), float(values[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ebm21cm.data.loreli_cache",
                                 description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--data", required=True)
    b.add_argument("--training-cache", required=True)
    b.add_argument("--run", required=True)
    b.add_argument("--split-file", required=True)
    b.add_argument("--l-perp", type=float, default=294.0)
    b.add_argument("--domain", type=int, default=DOMAIN)
    b.add_argument("--active", type=float, nargs=2, default=ACTIVE_RANGE)
    b.add_argument("--background", type=int, default=2)
    b.add_argument("--device", default="cuda")
    b.add_argument("--chunk", type=int, default=8)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--sims", type=int, nargs="*", default=None)
    b.add_argument("--parameter-mode", default="training-mean")
    b.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    build(a.out, a.data, a.training_cache, a.run, a.split_file, l_perp=a.l_perp,
          domain=a.domain, active=_parse_pair(a.active), background=a.background,
          device=a.device, chunk=a.chunk, limit=a.limit, sims=a.sims,
          parameter_mode=a.parameter_mode, overwrite=a.overwrite,
          log=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
