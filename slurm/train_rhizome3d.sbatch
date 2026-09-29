#!/bin/bash
# 3-D rhizome on fno-21cm's native LOS windows (one A100). Stops cleanly before the
# wall limit and, with CHAIN=N, queues N follow-up jobs (afterany) that resume
# from last.pt; a follow-up exits at once if the run already finished.
#   sbatch --export=ALL,RUN_DIR=runs/rhizome3d/w48,CHAIN=3,ARGS="--width 48 --rank 48" slurm/train_rhizome3d.sbatch
#SBATCH --job-name=rhz3d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rhz3d-%j.out
#SBATCH --error=logs/rhz3d-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh

RUN_DIR="${RUN_DIR:?set RUN_DIR}"
CHAIN="${CHAIN:-0}"
if [ -f "$RUN_DIR/test_metrics.json" ]; then
    echo "$RUN_DIR already finished"; exit 0
fi
if [ "$CHAIN" -gt 0 ]; then
    NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
        --export=ALL,RUN_DIR="$RUN_DIR",CHAIN=$((CHAIN - 1)),ARGS="${ARGS:-}" slurm/train_rhizome3d.sbatch)
    echo "queued follow-up $NEXT (CHAIN=$((CHAIN - 1)))"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# shellcheck disable=SC2086
python -m ebm21cm.train_rhizome3d --run-dir "$RUN_DIR" --workers $((SLURM_CPUS_PER_TASK - 2)) \
    --max-hours "${MAX_HOURS:-46.5}" ${ARGS:-}
