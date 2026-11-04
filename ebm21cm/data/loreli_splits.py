"""Parameter-block splits for LoReLi II.

The block is defined by the six-aligned grid index
``block = (sim_id - 10000) // 6``.  The 228 five-field cones form 60 blocks:
30 complete six-cone groups plus 30 partial groups (23 singletons, 2 pairs,
1 triple, 2 quadruples, 2 quintuples).  The corrected offset is essential:
the old ``(sim_id - 10001) // 6`` split real six-cone groups across blocks and
could leak near-duplicates between train and test.

The canonical v2 split stratifies the 30 full and 30 partial blocks separately
(22/4/4 each), giving 44 train / 8 validation / 8 test blocks and all 228
cones.  Train blocks are stored in a deterministic full/small interleave, so
the nested prefixes 4 ⊂ 8 ⊂ 16 ⊂ 32 ⊂ 44 each contain both group sizes.

The generator is :mod:`scripts.loreli_block_split`; the JSON schema is

    {"schema": "ebm21cm-loreli-blocksplit-v2",
     "seed": 0, "rule": "block = (sim_id - 10000) // 6", "offset": 10000,
     "splits": {"train": {"n_blocks": 44, "n_cones": ..., "blocks": {...}, "sims": [...]}, ...},
     "stage_slices": {"train": {"xHI_0.05-0.35": ...}, ...}}

Older hand-written schemas (top-level ``train``/``val``/``test`` lists or a
``blocks`` mapping) are still accepted for tests and legacy files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BLOCK_OFFSET = 10000
BLOCK_SIZE = 6
DEFAULT_SPLIT_FILE = "runs/loreli_splits/block_split_v2_seed0.json"
SPLIT_NAMES = ("train", "val", "test")
FULL_BLOCK_SIZE = 6
V2_SCHEMA = "ebm21cm-loreli-blocksplit-v2"


def block_of(sim_id: int) -> int:
    """Parameter block of a LoReLi simulation id (6-aligned from 10000)."""
    return (int(sim_id) - BLOCK_OFFSET) // BLOCK_SIZE


def block_for_split(sim_id: int, split: dict) -> int:
    """Block id using the offset recorded in the split file (legacy-safe)."""
    return (int(sim_id) - int(split.get("offset", BLOCK_OFFSET))) // int(
        split.get("block_size", BLOCK_SIZE)
    )


def sims_of_block(block: int, offset: int = BLOCK_OFFSET) -> list[int]:
    """Contiguous six grid simulations of a block (fallback when the file has no sim list)."""
    start = int(offset) + BLOCK_SIZE * int(block)
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
    by_block: dict[int, list[int]] = {}
    for sim in sims:
        by_block.setdefault(block_of(sim), []).append(sim)
    blocks = sorted(by_block)
    if len(blocks) != n_train + n_val + n_test:
        raise ValueError(
            f"expected {n_train + n_val + n_test} blocks, found {len(blocks)}"
        )
    order = np.random.default_rng(seed).permutation(blocks).tolist()
    split = {
        "seed": int(seed),
        "rule": "block = (sim_id - 10000) // 6",
        "offset": BLOCK_OFFSET,
        "block_size": BLOCK_SIZE,
        "train_blocks": order[:n_train],
        "val_blocks": order[n_train:n_train + n_val],
        "test_blocks": order[n_train + n_val:],
        "blocks": {},
    }
    for name in SPLIT_NAMES:
        for block in split[f"{name}_blocks"]:
            split["blocks"][str(block)] = {"split": name, "sims": sorted(by_block[block])}
    return split


def make_v2_split(sims, seed: int = 0, stage_slices: dict | None = None) -> dict:
    """Stratified 44/8/8 split over the 60 six-aligned blocks.

    The 30 complete blocks (6 cones) and the 30 incomplete blocks are each
    permuted with ``default_rng(seed)``; each split takes 22/4/4 of each kind.
    The train blocks are interleaved full/small, so every prefix used by the
    learning curve (4, 8, 16, 32, 44) contains both kinds.
    """
    import numpy as np

    sims = sorted({int(s) for s in sims})
    by_block: dict[int, list[int]] = {}
    for sim in sims:
        by_block.setdefault(block_of(sim), []).append(sim)
    for block in by_block:
        by_block[block] = sorted(by_block[block])
    full = sorted(b for b, v in by_block.items() if len(v) == FULL_BLOCK_SIZE)
    small = sorted(b for b, v in by_block.items() if len(v) != FULL_BLOCK_SIZE)
    if len(full) != 30 or len(small) != 30 or len(sims) != 228:
        raise ValueError(
            f"expected 30 full + 30 partial blocks and 228 cones, got "
            f"{len(full)} + {len(small)} and {len(sims)}"
        )
    rng_full = np.random.default_rng(seed)
    rng_small = np.random.default_rng(seed)
    full_order = [full[i] for i in rng_full.permutation(len(full))]
    small_order = [small[i] for i in rng_small.permutation(len(small))]

    def interleave(a, b):
        return [x for pair in zip(a, b) for x in pair]

    train_blocks = interleave(full_order[:22], small_order[:22])
    val_blocks = interleave(full_order[22:26], small_order[22:26])
    test_blocks = interleave(full_order[26:], small_order[26:])

    def entry(blocks):
        return {
            "n_blocks": len(blocks),
            "n_cones": int(sum(len(by_block[b]) for b in blocks)),
            "blocks": {str(b): by_block[b] for b in blocks},
            "sims": sorted(s for b in blocks for s in by_block[b]),
        }

    payload = {
        "schema": V2_SCHEMA,
        "seed": int(seed),
        "rule": "block = (sim_id - 10000) // 6",
        "block_rule_note": "30 complete 6-cone groups plus 30 partial groups; all 228 cones assigned",
        "offset": BLOCK_OFFSET,
        "block_size": BLOCK_SIZE,
        "n_cones": len(sims),
        "n_blocks": len(by_block),
        "n_blocks_used": len(train_blocks) + len(val_blocks) + len(test_blocks),
        "block_sizes": {
            str(size): sum(1 for value in by_block.values() if len(value) == size)
            for size in sorted({len(value) for value in by_block.values()})
        },
        "split_sizes": [len(train_blocks), len(val_blocks), len(test_blocks)],
        "train_blocks": list(train_blocks),
        "val_blocks": list(val_blocks),
        "test_blocks": list(test_blocks),
        "splits": {
            "train": entry(train_blocks),
            "val": entry(val_blocks),
            "test": entry(test_blocks),
        },
        "stage_slices": stage_slices,
    }
    return payload


def load_split(path: str | Path) -> dict:
    """Load and normalise a block-split JSON file."""
    path = Path(path)
    raw = json.loads(path.read_text())
    rule = str(raw.get("rule", ""))
    offset = int(raw.get("offset", 10001 if "10001" in rule else BLOCK_OFFSET))
    block_size = int(raw.get("block_size", BLOCK_SIZE))
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
            sims = value.get("sims", sims_of_block(block, offset))
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
            block_to_sims.setdefault(block, sims_of_block(block, offset))
    for block in block_to_split:
        block_to_sims.setdefault(block, sims_of_block(block, offset))

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
        "block_size": block_size,
        "offset": offset,
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
    """A nested prefix of the train-block order stored in the split file.

    The v2 generator stores the 44 train blocks in a deterministic full/small
    interleave; the prefixes 4, 8, 16, 32 and 44 therefore each contain both
    complete and partial parameter groups.  ``n <= 0`` or ``n >= 44`` means all
    training blocks.
    """
    blocks = list(split["train_blocks"])
    if n is None or int(n) <= 0 or int(n) >= len(blocks):
        return blocks
    return blocks[: int(n)]


def blocks_for_split(split: dict, name: str) -> list[int]:
    if name not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    return list(split[f"{name}_blocks"])


def filter_sims_by_split(sims, split: dict, name: str) -> list[int]:
    """Keep simulation ids that belong to ``name``, preserving input order."""
    allowed = set(split[f"{name}_sims"])
    return [int(sim) for sim in sims if int(sim) in allowed]
