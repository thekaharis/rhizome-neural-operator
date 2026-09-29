#!/bin/bash
# Long rhizome x_HI run on one A100, train/validation splits held in GPU memory
# (--data memory). Resubmitting with the same RUN_DIR and ARGS resumes from
# last.pt; the trainer refuses a resume whose options differ.
#   sbatch --export=ALL,RUN_DIR=runs/rhizome_xhi,ARGS="--width 64 --modes 24" slurm/train_rhizome.sbatch
#SBATCH --job-name=rhz-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=112G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rhz-train-%j.out
#SBATCH --error=logs/rhz-train-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh

CACHE="${CACHE:-$WORK/data/compressed/rhizome_xhi_slices.h5}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
RESUME=()
[ -f "$RUN_DIR/last.pt" ] && RESUME=(--resume)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# shellcheck disable=SC2086
python -m ebm21cm.train_rhizome --cache "$CACHE" --run-dir "$RUN_DIR" --device cuda --data memory \
    --threads "${SLURM_CPUS_PER_TASK:-4}" "${RESUME[@]}" ${ARGS:-}
