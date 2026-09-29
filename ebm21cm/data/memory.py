"""Cache splits held in memory, batched, normalized and augmented on the device.

:class:`~ebm21cm.data.cache.SliceDataset` reads and normalizes one row at a time
in Python. On 140x140 slices with 13 conditioning bands that starves a GPU.
:class:`MemorySplit` loads a split's stored float16 rows once (optionally only
some bands, or some cones) and builds whole batches on the device with the
same arithmetic as :class:`~ebm21cm.data.cache.Normalizer`, in float32.

Batches are dicts shaped like the ``DataLoader`` output of ``SliceDataset``
(``cond``, ``target`` with x_HI as ``2*x_HI-1`` in channel 0, ``scalars``, and
CPU ``row``/``cone_id``/``z``), so the existing training and evaluation code
consumes them unchanged. Only the x_HI target is loaded; ``target`` has one
channel.

Augmentation draws one transverse dihedral element per sample. Periodic rolls
are omitted: the recurrent operators commute with toroidal rolls exactly (up to
floating point), so rolls would not change what they learn.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import torch

from .cache import SCHEMA, SPLITS, Normalizer


def parse_channels(spec, bands) -> list[int]:
    """Band indices to keep: ``all``, ``center`` (the band holding the target
    slice), ``singles`` (every single-slice band), comma-separated indices, or
    an index sequence."""
    bands = [tuple(b) for b in bands]
    if spec in (None, "", "all"):
        return list(range(len(bands)))
    if spec == "center":
        return [j for j, (lo, hi) in enumerate(bands) if lo <= 0 < hi][:1]
    if spec == "singles":
        return [j for j, (lo, hi) in enumerate(bands) if hi - lo == 1]
    if isinstance(spec, str):
        out = [int(t) for t in spec.split(",") if t.strip()]
    else:
        out = [int(t) for t in spec]
    if not out or len(set(out)) != len(out) or any(not 0 <= j < len(bands) for j in out):
        raise ValueError(f"channels {spec!r} must be distinct indices into {len(bands)} bands")
    return out


def _runs(rows):
    """Contiguous [start, stop) runs of sorted row indices."""
    if len(rows) == 0:
        return []
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    starts = np.concatenate([[0], breaks])
    stops = np.concatenate([breaks, [len(rows)]])
    return [(int(rows[a]), int(rows[b - 1]) + 1) for a, b in zip(starts, stops)]


def dihedral(x, code):
    """Element ``code`` in 0..7 of the square's symmetry group on the last two axes."""
    x = torch.rot90(x, int(code) % 4, dims=(-2, -1))
    return x.flip(-1) if code >= 4 else x


class MemorySplit:
    """One split of a slice cache, resident on ``storage`` (a torch device)."""

    def __init__(self, path, split, stats, channels="all", cones=None, storage="cpu", device="cpu",
                 block_rows=512):
        self.path, self.split = str(path), split
        self.device, self.storage = torch.device(device), torch.device(storage)
        norm = Normalizer(stats)
        with h5py.File(self.path, "r") as h:
            if h.attrs.get("schema") != SCHEMA:
                raise ValueError(f"{path}: not an {SCHEMA} cache")
            rows = np.flatnonzero(h["split"][:] == SPLITS[split])
            cone = h["cone_id"][:]
            if cones is not None:
                rows = rows[np.isin(cone[rows], np.asarray(list(cones)))]
            self.bands = json.loads(h.attrs["bands"])
            self.channels = parse_channels(channels, self.bands)
            _, height, width = h["xhi"].shape
            delta = np.empty((len(rows), len(self.channels), height, width), np.float16)
            xhi = np.empty((len(rows), height, width), np.float16)
            pos = 0
            for start, stop in _runs(rows):
                for a in range(start, stop, block_rows):
                    b = min(stop, a + block_rows)
                    delta[pos:pos + b - a] = h["delta"][a:b][:, self.channels]
                    xhi[pos:pos + b - a] = h["xhi"][a:b]
                    pos += b - a
            z, params = h["z"][:][rows], h["params"][:][rows]
            self.rows, self.cone_id, self.z = rows, cone[rows], z
        center = [j for j, (lo, hi) in enumerate(self.bands) if lo <= 0 < hi]
        self.center_band = self.channels.index(center[0]) if center and center[0] in self.channels else 0
        self.delta = torch.from_numpy(delta).to(self.storage)
        self.xhi = torch.from_numpy(xhi).to(self.storage)
        self.scalars = torch.from_numpy(norm.scalars(z, params).astype(np.float32)).to(self.storage)
        self._mean = float(stats["delta_log1p_mean"])
        self._std = float(stats["delta_log1p_std"])
        self._rows_t = torch.from_numpy(rows.astype(np.int64))
        self._cone_t = torch.from_numpy(self.cone_id.astype(np.int64))
        self._z_t = torch.from_numpy(z.astype(np.float64))

    def __len__(self):
        return len(self.rows)

    @property
    def n_cond(self):
        return len(self.channels)

    @property
    def n_scalars(self):
        return self.scalars.shape[1]

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.delta, self.xhi, self.scalars))

    def batch(self, positions, codes=None):
        """Normalized batch at split ``positions``; optional per-sample dihedral ``codes``."""
        positions = torch.as_tensor(positions, dtype=torch.int64)
        on = positions.to(self.storage)
        delta = self.delta[on].to(self.device, non_blocking=True).float()
        cond = (torch.log1p(delta.clamp_min(-0.99)) - self._mean) / self._std
        target = 2 * self.xhi[on].to(self.device, non_blocking=True).float()[:, None] - 1
        if codes is not None:
            codes = torch.as_tensor(codes)
            for code in codes.unique().tolist():
                pick = (codes == code).nonzero()[:, 0].to(self.device)
                cond[pick] = dihedral(cond[pick], code)
                target[pick] = dihedral(target[pick], code)
        return {
            "cond": cond, "target": target, "scalars": self.scalars[on].to(self.device, non_blocking=True),
            "row": self._rows_t[positions], "cone_id": self._cone_t[positions], "z": self._z_t[positions],
        }

    def loader(self, batch_size, positions=None):
        """Re-iterable, unaugmented batches in split order (or over ``positions``)."""
        return _Loader(self, batch_size, positions)


class _Loader:
    def __init__(self, split, batch_size, positions):
        self.split, self.batch_size = split, batch_size
        self.positions = torch.arange(len(split)) if positions is None else torch.as_tensor(positions)

    def __len__(self):
        return -(-len(self.positions) // self.batch_size)

    def __iter__(self):
        for a in range(0, len(self.positions), self.batch_size):
            yield self.split.batch(self.positions[a:a + self.batch_size])


class MemoryStream:
    """Shuffled, dihedrally augmented batches with exact order/augmentation resumption.

    Same state layout as ``train_rhizome.TrainingStream``; one CPU generator
    drives both the permutation and the per-sample augmentation codes.
    """

    def __init__(self, split, batch_size, seed):
        self.dataset, self.batch_size = split, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(split), generator=self.generator)
        self.position, self.epoch = 0, 0

    def next(self):
        if self.position == len(self.order):
            self.order = torch.randperm(len(self.dataset), generator=self.generator)
            self.position, self.epoch = 0, self.epoch + 1
        end = min(self.position + self.batch_size, len(self.order))
        positions = self.order[self.position:end]
        codes = torch.randint(0, 8, (len(positions),), generator=self.generator)
        self.position = end
        return self.dataset.batch(positions, codes)

    def state_dict(self):
        return {"order": self.order, "position": self.position, "epoch": self.epoch,
                "generator": self.generator.get_state(), "augmentation": None}

    def load_state_dict(self, state):
        self.order, self.position, self.epoch = state["order"], state["position"], state["epoch"]
        self.generator.set_state(state["generator"])
