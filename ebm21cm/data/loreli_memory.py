"""Split/stream for the LoReLi fine-tuning cache, host RAM or disk-backed.

The cache row is ~4.7 MB (26x280x280 float16 conditioning, 2x280x280 float16
target, 280x280 float32 physical factor).  It stores every target slice whose
mean x_HI is in [0.02, 0.98] plus two background slices, which is 292-382 rows
per cone rather than the ~200 stage slices of the evaluation.  The v2 split has
44 training blocks (170 cones), so the 44-block train+val set is ~272-310 GB
and does not fit the 240 GB node request.

``storage="cpu"`` (the default) loads the selected rows into host RAM exactly as
before.  ``storage="disk"`` keeps only the scalars, mask and row metadata in RAM
and reads ``cond``/``target``/``phys`` from the HDF5 cache for each batch.  The
batch's file rows are sorted and read as contiguous runs, so a batch costs at
most ``batch_size`` HDF5 reads per dataset instead of one per row; HDF5's own
chunk cache and the OS page cache provide the read-ahead.  No cross-batch chunk
buffer is kept: with a global random permutation a 32-row buffer would be hit
by ~2.5% of batches, which does not pay for the RAM.

Shuffling is the same exact global permutation in both modes
(:class:`LoReLiStream` permutes split positions, never cache chunks), and the
returned tensors are bit-identical: float16 storage values are exactly
representable in float32, so reading a row from disk and reading it from RAM
give the same batch, augmentation codes and loss.  A disk run is therefore
directly comparable with the 8-block cpu arms.

Dihedral augmentation is applied to the conditioning, target and physical
factor together.  The transverse mask is a centred square, invariant under the
dihedral group, but it is transformed with the same code for consistency.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch

from .loreli_cache import SCHEMA
from .memory import _Loader, dihedral


class LoReLiMemorySplit:
    """One split (optionally a subset of training blocks) on ``storage``.

    ``storage`` is either a torch device (``"cpu"``, ``"cuda"``, ...), which
    keeps the selected rows resident there, or the string ``"disk"``, which
    reads them per batch.
    """

    def __init__(
        self,
        path,
        split: str,
        *,
        blocks=None,
        storage="cpu",
        device="cpu",
        block_rows: int = 128,
    ):
        self.path, self.split = str(path), split
        self.device = torch.device(device)
        self.storage_mode = "disk" if str(storage) == "disk" else "memory"
        self.storage = (
            torch.device("cpu") if self.storage_mode == "disk" else torch.device(storage)
        )
        with h5py.File(self.path, "r") as h:
            if h.attrs.get("schema") != SCHEMA:
                raise ValueError(f"{path}: not an {SCHEMA} cache")
            self.attrs = dict(h.attrs)
            split_id = {"train": 0, "val": 1, "test": 2}[split]
            rows = np.flatnonzero(h["split"][:] == split_id)
            if blocks is not None:
                wanted = np.asarray(list(blocks), dtype=np.int32)
                rows = rows[np.isin(h["block"][:][rows], wanted)]
            if rows.size == 0:
                raise ValueError(f"{path}: split {split!r} with blocks {blocks} is empty")
            self.row_bytes = int(sum(
                np.prod(h[key].shape[1:], dtype=np.int64) * h[key].dtype.itemsize
                for key in ("cond", "target", "phys")
            ))
            arrays = None
            if self.storage_mode == "memory":
                arrays = {
                    key: np.empty((len(rows), *h[key].shape[1:]), h[key].dtype)
                    for key in ("cond", "target", "phys")
                }
                pos = 0
                for start, stop in _runs(rows):
                    for a in range(start, stop, block_rows):
                        b = min(stop, a + block_rows)
                        for key, array in arrays.items():
                            array[pos:pos + b - a] = h[key][a:b]
                        pos += b - a
            self.scalars = torch.from_numpy(h["scalars"][:][rows].astype(np.float32)).to(self.storage)
            self.mask = torch.from_numpy(h["mask"][:].astype(bool)).to(self.storage)
            self.z = h["z"][:][rows]
            self.cone_id = h["cone_id"][:][rows]
            self.block = h["block"][:][rows]
            self.rows = rows
            self.n_out = int(h.attrs["n_out"])
            self.domain = int(h.attrs["domain"])
        if self.storage_mode == "disk":
            self.cond = self.target = self.phys = None
            self._disk = _DiskRows(self.path, self.row_bytes)
        else:
            self._disk = None
            self.cond = torch.from_numpy(arrays["cond"]).to(self.storage)
            self.target = torch.from_numpy(arrays["target"]).to(self.storage)
            self.phys = torch.from_numpy(arrays["phys"]).to(self.storage)
        self._z_t = torch.from_numpy(self.z.astype(np.float64))
        self._cone_t = torch.from_numpy(self.cone_id.astype(np.int64))

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def nbytes(self) -> int:
        """Bytes resident on ``storage`` (scalars/mask always; rows only for cpu/device)."""
        tensors = [self.scalars, self.mask]
        tensors += [t for t in (self.cond, self.target, self.phys) if t is not None]
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @property
    def dataset_nbytes(self) -> int:
        """Bytes of ``cond``/``target``/``phys`` for the selected rows, resident or not."""
        return len(self.rows) * self.row_bytes

    @property
    def io_stats(self) -> dict:
        """Disk read counters; zeros when the split is RAM-resident."""
        if self._disk is None:
            return {"reads": 0, "bytes_read": 0}
        return {"reads": self._disk.reads, "bytes_read": self._disk.bytes_read}

    def close(self) -> None:
        if self._disk is not None:
            self._disk.close()

    def batch(self, positions, codes=None):
        positions = torch.as_tensor(positions, dtype=torch.int64)
        on = positions.to(self.storage)
        if self.storage_mode == "disk":
            cond_np, target_np, phys_np = self._disk.read(self.rows[positions.cpu().numpy()])
            cond = torch.from_numpy(cond_np).to(self.device, non_blocking=True).float()
            target = torch.from_numpy(target_np).to(self.device, non_blocking=True).float()
            phys = torch.from_numpy(phys_np).to(self.device, non_blocking=True).float()
        else:
            get = lambda tensor: tensor[on].to(self.device, non_blocking=True).float()
            cond, target, phys = get(self.cond), get(self.target), get(self.phys)
        mask = self.mask.to(self.device)[None, None].expand(len(positions), 1, self.domain, self.domain).clone()
        if codes is not None:
            codes = torch.as_tensor(codes)
            for code in codes.unique().tolist():
                pick = (codes == code).nonzero()[:, 0].to(self.device)
                cond[pick], target[pick], phys[pick], mask[pick] = (
                    dihedral(t[pick], int(code)) for t in (cond, target, phys, mask)
                )
        return {
            "cond": cond,
            "target": target,
            "phys": phys[:, None],
            "mask": mask,
            "scalars": self.scalars[on].to(self.device, non_blocking=True),
            "cone_id": self._cone_t[positions],
            "z": self._z_t[positions],
        }

    def loader(self, batch_size, positions=None):
        return _Loader(self, batch_size, positions)


class _DiskRows:
    """Per-batch reader for the HDF5 cache, one contiguous run at a time.

    ``rows`` are absolute cache row indices.  A batch is read in file order so
    adjacent rows share one HDF5 call; the result is scattered back into the
    batch's original order.  One open read-only handle is kept for the run.
    """

    def __init__(self, path, row_bytes):
        self.path = str(path)
        self.row_bytes = int(row_bytes)
        self._handle = None
        self.reads = 0
        self.bytes_read = 0

    def handle(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def read(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        h = self.handle()
        cond = np.empty((rows.size, *h["cond"].shape[1:]), h["cond"].dtype)
        target = np.empty((rows.size, *h["target"].shape[1:]), h["target"].dtype)
        phys = np.empty((rows.size, *h["phys"].shape[1:]), h["phys"].dtype)
        order = np.argsort(rows, kind="stable")
        ordered = rows[order]
        start = 0
        while start < ordered.size:
            stop = start + 1
            while stop < ordered.size and ordered[stop] == ordered[stop - 1] + 1:
                stop += 1
            first, last = int(ordered[start]), int(ordered[stop - 1]) + 1
            take = order[start:stop]
            cond[take] = h["cond"][first:last]
            target[take] = h["target"][first:last]
            phys[take] = h["phys"][first:last]
            self.reads += 1
            self.bytes_read += (last - first) * self.row_bytes
            start = stop
        return cond, target, phys

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _runs(rows):
    if len(rows) == 0:
        return []
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    starts = np.concatenate([[0], breaks])
    stops = np.concatenate([breaks, [len(rows)]])
    return [(int(rows[a]), int(rows[b - 1]) + 1) for a, b in zip(starts, stops)]


class LoReLiStream:
    """Shuffled, dihedrally augmented batches with resumable order/state."""

    def __init__(self, split: LoReLiMemorySplit, batch_size: int, seed: int):
        self.dataset, self.batch_size = split, int(batch_size)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.order = torch.randperm(len(split), generator=self.generator)
        self.position = 0

    def next(self):
        if self.position == len(self.order):
            self.order = torch.randperm(len(self.dataset), generator=self.generator)
            self.position = 0
        positions = self.order[self.position:self.position + self.batch_size]
        codes = torch.randint(0, 8, (len(positions),), generator=self.generator)
        self.position += len(positions)
        return self.dataset.batch(positions, codes)

    def state_dict(self):
        return {
            "order": self.order,
            "position": self.position,
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state):
        self.order = state["order"]
        self.position = int(state["position"])
        self.generator.set_state(state["generator"])
