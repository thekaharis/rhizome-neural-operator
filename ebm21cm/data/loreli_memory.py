"""In-memory split/stream for the LoReLi fine-tuning cache.

The cache is large (about 4.5 MB per 280x280 slice), so the default storage is
host RAM.  With the full 28 training blocks the training split is ~150 GB,
which fits a 240 GB node; the 8/16-block arms are much smaller.  Batches copy
one batch to the GPU and cast to float32.

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
    """One split (optionally a subset of training blocks) resident on ``storage``."""

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
        self.device, self.storage = torch.device(device), torch.device(storage)
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
        self.cond = torch.from_numpy(arrays["cond"]).to(self.storage)
        self.target = torch.from_numpy(arrays["target"]).to(self.storage)
        self.phys = torch.from_numpy(arrays["phys"]).to(self.storage)
        self._z_t = torch.from_numpy(self.z.astype(np.float64))
        self._cone_t = torch.from_numpy(self.cone_id.astype(np.int64))

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.cond, self.target, self.phys, self.scalars, self.mask)
        )

    def batch(self, positions, codes=None):
        positions = torch.as_tensor(positions, dtype=torch.int64)
        on = positions.to(self.storage)
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
