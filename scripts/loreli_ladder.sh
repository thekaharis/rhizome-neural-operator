#!/bin/bash
# One LoReLi fine-tuning ladder: one block count x {decoder, ends, pointwise, none, scratch},
# each arm followed by a test-split evaluation (afterok).  Run from the repository root.
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
# Environment (all optional):
#   CACHE    slice cache                     (default: data/compressed/loreli_slices_l294_v3.h5)
#   SCALARS  history side file, or "none" for the cache's own (oracle) histories
#                                            (default: data/compressed/loreli_slices_l294_v3_emuhist.h5)
#   TAG      run-name prefix                 (default: l294v3)
#   ARMS     space-separated subset of the five arms
#   SEED     (default 0)   STEPS (12000)   WALL (04:00:00)
#   GPU/MEM  default by storage: disk -> a100 / 240G; cpu -> 8: a100 / 240G (~62 GiB train+val for v3),
#            16: a100 / 340G, 32/44: h200 / 520G (44 blocks ~423 GiB in RAM)
#   DEPEND   a job id; every training job waits for it (afterok)
#
# Running this submits up to ten jobs (five GPU trainings, five short GPU evaluations).
set -euo pipefail

BLOCKS="${1:?usage: scripts/loreli_ladder.sh 8|16|32|44}"
case "$BLOCKS" in 8|16|32|44) ;; *) echo "BLOCKS must be 8, 16, 32 or 44" >&2; exit 2 ;; esac

# cpu is the default; 32/44 blocks do not fit host RAM and use disk.
if [ "$BLOCKS" -ge 32 ]; then DEFAULT_STORAGE=disk; else DEFAULT_STORAGE=cpu; fi
STORAGE="${STORAGE:-$DEFAULT_STORAGE}"

WORK=/pfs/10/work/hd_id260-fno_training
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # repository root
CACHE="${CACHE:-$WORK/data/compressed/loreli_slices_l294_v3.h5}"
SCALARS="${SCALARS:-$WORK/data/compressed/loreli_slices_l294_v3_emuhist.h5}"
TAG="${TAG:-l294v3}"
RUN="${RUN:-$WORK/rhizome-neural-operator/runs/rhizome_mf2d_w64_structured_all}"
SPLIT="${SPLIT:-$ROOT/runs/loreli_splits/block_split_v2_seed0.json}"
OUT="${OUT:-$ROOT/runs/loreli_finetune}"
STEPS="${STEPS:-12000}"
WALL="${WALL:-04:00:00}"
SEED="${SEED:-0}"
ARMS="${ARMS:-decoder ends pointwise none scratch}"
if [ "$STORAGE" = disk ]; then GPU="${GPU:-a100}"; MEM="${MEM:-240G}"
else
    case "$BLOCKS" in
        32|44) GPU="${GPU:-h200}"; MEM="${MEM:-520G}" ;;
        16)    GPU="${GPU:-a100}"; MEM="${MEM:-340G}" ;;
        *)     GPU="${GPU:-a100}"; MEM="${MEM:-240G}" ;;
    esac
fi

[ -f "$SPLIT" ] || { echo "missing split: $SPLIT" >&2; exit 2; }
if [ -z "${DEPEND:-}" ]; then
    [ -f "$CACHE" ] || { echo "missing cache: $CACHE" >&2; exit 2; }
    if [ "$SCALARS" != none ]; then [ -f "$SCALARS" ] || { echo "missing side file: $SCALARS" >&2; exit 2; }; fi
fi
if [ "$SCALARS" = none ]; then HIST=oracle; SIDE_ARG=""; else HIST=emu; SIDE_ARG="--scalars-file $SCALARS"; fi
DEP_ARG=""; [ -n "${DEPEND:-}" ] && DEP_ARG="--dependency=afterok:$DEPEND"

for arm in $ARMS; do
    case "$arm" in decoder|ends|pointwise|none|scratch) ;; *) echo "unknown arm $arm" >&2; exit 2 ;; esac
    freeze="$arm"; extra=""
    if [ "$arm" = scratch ]; then freeze=none; extra="--scratch"; fi
    dir="$OUT/${TAG}_b${BLOCKS}_${arm}_${HIST}_seed${SEED}"
    args="--cache $CACHE --run-dir $dir --init-from $RUN --split-file $SPLIT --freeze $freeze $extra \
        --train-blocks $BLOCKS --seed $SEED --device cuda --storage $STORAGE --batch-size 8 --max-steps $STEPS --min-steps 1500 \
        --val-every 200 --lr-new 1e-3 --lr-pretrained 1e-4 $SIDE_ARG"
    # shellcheck disable=SC2086
    job=$(sbatch --parsable $DEP_ARG --time="$WALL" --mem="$MEM" --gres="gpu:$GPU:1" \
        --job-name="ft-b${BLOCKS}-${arm}" --export=ALL,ARGS="$args" slurm/train_loreli_finetune.sbatch)
    eargs="--run $dir --cache $CACHE --split test"
    ev=$(sbatch --parsable --dependency=afterok:"$job" --job-name="ev-b${BLOCKS}-${arm}" \
        --export=ALL,ARGS="$eargs" slurm/evaluate_loreli_finetune.sbatch)
    echo "$arm: train $job -> $dir; test eval $ev"
done
