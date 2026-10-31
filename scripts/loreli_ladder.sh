#!/bin/bash
# Submit one rung of the LoReLi fine-tuning ladder:
# one block count x {decoder, ends, pointwise, none, scratch}.
# Run from the handoff worktree.
#
#   scripts/loreli_ladder.sh 8      # storage=cpu
#   scripts/loreli_ladder.sh 16     # storage=cpu
#   scripts/loreli_ladder.sh 32     # storage=disk (overridable with STORAGE=cpu)
#   scripts/loreli_ladder.sh 44     # storage=disk (overridable with STORAGE=cpu)
#
# storage=cpu loads the split into host RAM and is the default everywhere.
# storage=disk reads each batch from the HDF5 cache and keeps only the scalars
# and mask in RAM; the 32- and 44-block arms need it because their train+val
# rows (~218 GB and ~272-310 GB at 292-382 rows/cone) do not fit the 240 GB
# node request.  Both modes use the same exact global row permutation, so batch
# composition is identical and the rungs stay comparable.
#
# No job is launched by the code that produced this script; running it submits
# five A100 jobs of up to four hours each.
set -euo pipefail

BLOCKS="${1:?usage: scripts/loreli_ladder.sh 8|16|32|44}"
case "$BLOCKS" in 8|16|32|44) ;; *) echo "BLOCKS must be 8, 16, 32 or 44" >&2; exit 2 ;; esac

# cpu is the default; 32/44 blocks do not fit host RAM and use disk.
if [ "$BLOCKS" -ge 32 ]; then DEFAULT_STORAGE=disk; else DEFAULT_STORAGE=cpu; fi
STORAGE="${STORAGE:-$DEFAULT_STORAGE}"

ROOT="${ROOT:-/pfs/10/work/hd_id260-fno_training/handoff/rhizome-loreli}"
CACHE="${CACHE:-$ROOT/runs/loreli_cache/loreli_slices_l294_v2.h5}"
RUN="${RUN:-/pfs/10/work/hd_id260-fno_training/rhizome-neural-operator/runs/rhizome_mf2d_w64_structured_all}"
SPLIT="${SPLIT:-$ROOT/runs/loreli_splits/block_split_v2_seed0.json}"
OUT="${OUT:-$ROOT/runs/loreli_finetune}"
STEPS="${STEPS:-12000}"
WALL="${WALL:-04:00:00}"

[ -f "$CACHE" ] || { echo "missing cache: $CACHE" >&2; exit 2; }
[ -f "$SPLIT" ] || { echo "missing split: $SPLIT" >&2; exit 2; }

for arm in decoder ends pointwise none; do
    args="--cache $CACHE --run-dir $OUT/l294_b${BLOCKS}_${arm}_seed0 \
        --init-from $RUN --split-file $SPLIT --freeze $arm --train-blocks $BLOCKS \
        --seed 0 --device cuda --storage $STORAGE --batch-size 8 \
        --max-steps $STEPS --min-steps 1500 --val-every 200 \
        --lr-new 1e-3 --lr-pretrained 1e-4"
    sbatch --time="$WALL" --export=ALL,ARGS="$args" slurm/train_loreli_finetune.sbatch
done

args="--cache $CACHE --run-dir $OUT/l294_b${BLOCKS}_scratch_seed0 \
    --init-from $RUN --split-file $SPLIT --freeze none --scratch --train-blocks $BLOCKS \
    --seed 0 --device cuda --storage $STORAGE --batch-size 8 \
    --max-steps $STEPS --min-steps 1500 --val-every 200 \
    --lr-new 1e-3 --lr-pretrained 1e-4"
sbatch --time="$WALL" --export=ALL,ARGS="$args" slurm/train_loreli_finetune.sbatch
