"""Parameter-block splits for LoReLi II.

The block is defined by the grid index ``block = (sim_id - 10001) // 6``.  Only
228 of the 9,828 grid points have five-field files, so the nonempty blocks are
irregular (95 of them in the current data, some with a single cone).  Binac's
canonical split is a seed-0 permutation of the nonempty blocks, taking 28 train,
5 validation and 5 test blocks; the remaining 57 blocks/136 cones are unused.

Cones within one block differ only in the X-ray spectral index, so splitting by
cone would leak near-duplicates across train/validation/test.

The JSON file is written by Binac; this module accepts the explicit schema

    {"seed": 0, "block_size": 6, "offset": 10001,
     "train_blocks": [...], "val_blocks": [...], "test_blocks": [...],
     "blocks": {"0": {"split": "train", "sims": [10001, ...]}, ...}}

and the common shorthand where ``train``/``val``/``test`` are block lists or
simulation lists, as well as the current ``splits`` schema::

    {"splits": {"train": {"blocks": {"391": [12352], ...}, "sims": [...]}, ...}}
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BLOCK_OFFSET = 10001
BLOCK_SIZE = 6
DEFAULT_SPLIT_FILE = "runs/loreli_splits/block_split_seed0.json"
SPLIT_NAMES = ("train", "val", "test")


def block_of(sim_id: int) -> int:
    """Parameter block of a LoReLi simulation id."""
    return (int(sim_id) - BLOCK_OFFSET) // BLOCK_SIZE


def sims_of_block(block: int) -> list[int]:
    """Contiguous six grid simulations of a block (fallback when the file has no sim list)."""
    start = BLOCK_OFFSET + BLOCK_SIZE * int(block)
    return list(range(start, start + BLOCK_SIZE))


def _as_blocks(values) -> list[int]:
    """Interpret a split list as block ids (all values < offset) or sim ids."""
    values = [int(v) for v in values]
    if any(v >= BLOCK_OFFSET for v in values):
        return sorted({block_of(v) for v in values})
    return values


def make_split(sims, seed: int = 0, n_train: int = 28, n_val: int = 5, n_test: int = 5) -> dict:
    """Synthetic contiguous-block split, for tests and when no external file exists."""
    import numpy as np

    sims = sorted({int(s) for s in sims})
    blocks = sorted({block_of(s) for s in sims})
    if len(blocks) != n_train + n_val + n_test:
        raise ValueError(
            f"expected {n_train + n_val + n_test} blocks, found {len(blocks)}"
        )
    order = np.random.default_rng(seed).permutation(blocks).tolist()
    split = {
        "seed": int(seed),
        "block_size": BLOCK_SIZE,
        "offset": BLOCK_OFFSET,
        "train_blocks": order[:n_train],
        "val_blocks": order[n_train:n_train + n_val],
        "test_blocks": order[n_train + n_val:],
        "blocks": {},
    }
    for name in SPLIT_NAMES:
        for block in split[f"{name}_blocks"]:
            split["blocks"][str(block)] = {"split": name, "sims": sims_of_block(block)}
    return split


def load_split(path: str | Path) -> dict:
    """Load and normalise a block-split JSON file."""
    path = Path(path)
    raw = json.loads(path.read_text())
    block_to_split: dict[int, str] = {}
    block_to_sims: dict[int, list[int]] = {}
    split_blocks: dict[str, list[int]] = {}

    splits = raw.get("splits")
    if isinstance(splits, dict):
        alias = {"validation": "val"}
        for key, node in splits.items():
            name = alias.get(str(key), str(key))
            if name not in SPLIT_NAMES or not isinstance(node, dict):
                continue
            node_blocks = node.get("blocks") or {}
            if isinstance(node_blocks, dict):
                split_blocks.setdefault(name, [])
                for block_key, sims in node_blocks.items():
                    block = int(block_key)
                    block_to_sims[block] = sorted(int(s) for s in sims)
                    block_to_split[block] = name
                    split_blocks[name].append(block)
            elif isinstance(node_blocks, (list, tuple)):
                split_blocks.setdefault(name, [])
                for block in node_blocks:
                    block_to_split[int(block)] = name
                    split_blocks[name].append(int(block))

    blocks = raw.get("blocks") or {}
    for key, value in blocks.items():
        block = int(key)
        if isinstance(value, dict):
            split = value.get("split")
            sims = value.get("sims", sims_of_block(block))
        else:
            split = None
            sims = value
        block_to_sims[block] = sorted(int(s) for s in sims)
        if split in SPLIT_NAMES:
            block_to_split[block] = split
        elif split in ("validation",):
            block_to_split[block] = "val"

    for name in SPLIT_NAMES:
        values = raw.get(f"{name}_blocks")
        if values is None:
            values = raw.get(name)
        if values is None and name == "val":
            values = raw.get("validation_blocks", raw.get("validation"))
        if values is not None:
            split_blocks[name] = _as_blocks(values)
    if not split_blocks:
        split_blocks = {
            name: [b for b, s in block_to_split.items() if s == name]
            for name in SPLIT_NAMES
        }
    if not split_blocks:
        raise ValueError(
            f"{path}: no train/val/test block lists; expected train_blocks/val_blocks/test_blocks"
        )

    for name in SPLIT_NAMES:
        for block in split_blocks.get(name, []):
            if block in block_to_split and block_to_split[block] != name:
                raise ValueError(f"{path}: block {block} appears in both {block_to_split[block]} and {name}")
            block_to_split[block] = name
            block_to_sims.setdefault(block, sims_of_block(block))
    for block in block_to_split:
        block_to_sims.setdefault(block, sims_of_block(block))

    assigned = set(block_to_split)
    duplicates = [
        block for block in assigned if sum(block in split_blocks.get(n, []) for n in SPLIT_NAMES) > 1
    ]
    if duplicates:
        raise ValueError(f"{path}: blocks assigned more than once: {duplicates}")

    out = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed": int(raw.get("seed", 0)),
        "block_size": int(raw.get("block_size", BLOCK_SIZE)),
        "offset": int(raw.get("offset", BLOCK_OFFSET)),
        "train_blocks": split_blocks.get("train", []),
        "val_blocks": split_blocks.get("val", []),
        "test_blocks": split_blocks.get("test", []),
        "block_to_split": {int(k): v for k, v in block_to_split.items()},
        "block_to_sims": {int(k): list(v) for k, v in block_to_sims.items()},
    }
    out["train_sims"] = sorted(s for b in out["train_blocks"] for s in out["block_to_sims"].get(b, []))
    out["val_sims"] = sorted(s for b in out["val_blocks"] for s in out["block_to_sims"].get(b, []))
    out["test_sims"] = sorted(s for b in out["test_blocks"] for s in out["block_to_sims"].get(b, []))
    out["sim_to_block"] = {
        sim: block for block, sims in out["block_to_sims"].items() for sim in sims
    }
    out["sim_to_split"] = {
        sim: out["block_to_split"][block]
        for block, sims in out["block_to_sims"].items()
        if block in out["block_to_split"]
        for sim in sims
    }
    return out


def nested_train_blocks(split: dict, n: int | None) -> list[int]:
    """A deterministic nested prefix of a seed-based shuffle of the train blocks.

    The external file's block mapping is sorted by block id, so taking its first
    ``n`` blocks would confound the learning-curve arms with parameter-grid
    position.  We instead shuffle the train block ids with the split's seed;
    the 4/8/16/28 subsets are prefixes of one permutation and therefore nested.
    ``n <= 0`` or ``n >= len(train_blocks)`` means all training blocks.
    """
    blocks = list(split["train_blocks"])
    if n is None or int(n) <= 0:
        n = len(blocks)
    import numpy as np

    order = np.random.default_rng(int(split.get("seed", 0))).permutation(len(blocks))
    return [blocks[int(i)] for i in order[: min(int(n), len(blocks))]]


def blocks_for_split(split: dict, name: str) -> list[int]:
    if name not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    return list(split[f"{name}_blocks"])


def filter_sims_by_split(sims, split: dict, name: str) -> list[int]:
    """Keep simulation ids that belong to ``name``, preserving input order."""
    allowed = set(split[f"{name}_sims"])
    return [int(sim) for sim in sims if int(sim) in allowed]
