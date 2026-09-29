#!/bin/bash
# Train on one A100. Resubmitting with the same RUN_DIR resumes from last.pt.
#   sbatch --export=ALL,RUN_DIR=runs/base slurm/train.sbatch
#SBATCH --job-name=ebm-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ebm-train-%j.out
#SBATCH --error=logs/ebm-train-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh

CACHE="${CACHE:-$WORK/data/compressed/ebm_slices.h5}"
RUN_DIR="${RUN_DIR:-runs/base}"
CONFIG="${CONFIG:-configs/base.json}"
RESUME=()
[ -f "$RUN_DIR/last.pt" ] && RESUME=(--resume)
# shellcheck disable=SC2086
python -m ebm21cm.train --cache "$CACHE" --run-dir "$RUN_DIR" --config "$CONFIG" \
    --device cuda "${RESUME[@]}" ${OVERRIDES:+--set $OVERRIDES}
