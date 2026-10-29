"""Generate the corrected LoReLi v2 block split.

The six-cone r_H/S groups align at multiples of 6 from simulation id **10000**:
``block = (sim_id - 10000) // 6``.  The 228 five-field cones form 60 blocks:
30 complete groups of six plus 30 partial groups (23 singletons, 2 pairs,
1 triple, 2 quadruples, 2 quintuples).

The split is stratified by group size: the 30 full and 30 partial blocks are
each permuted with ``default_rng(seed)`` and split 22 / 4 / 4, giving 44 train /
8 validation / 8 test blocks and all 228 cones.  Train blocks are interleaved
full/small, so the learning-curve prefixes 4/8/16/32/44 each contain both kinds.

Stage-slice counts are read from the per-cone ``n_active`` records of the saved
evaluation (`--pred`), which use the same border-4 / LOS-margin-28 / truth-mean
selection as :mod:`ebm21cm.evaluate_loreli`.

    python scripts/loreli_block_split.py \
        --pred runs/eval3d/loreli_lc128/rhizome_mf2d_w64_structured_all_full_v2 \
        --out runs/loreli_splits/block_split_v2_seed0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ebm21cm.data.loreli_splits import V2_SCHEMA, make_v2_split

STAGES = ("xHI_0.05-0.35", "xHI_0.35-0.65", "xHI_0.65-0.95")


def read_sims(pred: Path) -> list[int]:
    sims = sorted(int(p.stem.split("_")[-1]) for p in pred.glob("cone_*.h5"))
    if not sims:
        raise SystemExit(f"{pred}: no cone_*.h5 files")
    if len(set(sims)) != len(sims):
        raise SystemExit(f"{pred}: duplicate simulation numbers")
    return sims


def stage_counts(pred: Path, sims) -> tuple[dict, list[int]]:
    """Sum per-cone `n_active` by stage; track cones without a record."""
    counts = {stage: 0 for stage in STAGES}
    missing = []
    for sim in sims:
        path = pred / f"cone_{sim}.json"
        if not path.exists():
            missing.append(int(sim))
            continue
        record = json.loads(path.read_text())
        active = record.get("n_active", {})
        for stage in STAGES:
            counts[stage] += int(active.get(stage, 0))
    return counts, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pred", type=Path, required=True,
                    help="directory with cone_*.h5 and cone_*.json from a full evaluation")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    sims = read_sims(args.pred)
    if len(sims) != 228:
        raise SystemExit(f"{args.pred}: expected 228 cones, found {len(sims)}")
    payload = make_v2_split(sims, seed=args.seed)

    stage_slices = {}
    for name in ("train", "val", "test"):
        counts, missing = stage_counts(args.pred, payload["splits"][name]["sims"])
        stage_slices[name] = {stage: int(value) for stage, value in counts.items()}
        if missing:
            print(f"warning: {name}: {len(missing)} cones have no n_active record", flush=True)
    payload["stage_slices"] = stage_slices
    payload["stage_slices_note"] = (
        "sum of per-cone n_active from cone_*.json; border 4, LOS margin 28, "
        "stage by truth slice-mean x_HI"
    )
    payload["pred"] = str(args.pred.resolve())
    payload["schema"] = V2_SCHEMA

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"{len(sims)} cones, {payload['n_blocks']} blocks, sizes {payload['block_sizes']}", flush=True)
    for name in ("train", "val", "test"):
        entry = payload["splits"][name]
        stages = stage_slices[name]
        print(f"{name:5s} blocks {entry['n_blocks']:2d} cones {entry['n_cones']:3d}  "
              f"slices {stages['xHI_0.05-0.35']} / {stages['xHI_0.35-0.65']} / "
              f"{stages['xHI_0.65-0.95']}", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
