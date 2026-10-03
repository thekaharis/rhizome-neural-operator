#!/bin/bash
# Multi-field 2-D slice-wise rhizome (ebm21cm.train_rhizome_mf2d) on one A100, splits in host RAM.
# Resubmitting with the same RUN_DIR and ARGS resumes from last.pt.
#   ARGS="--width 64 --batch-size 32 --head structured" sbatch --export=ALL,RUN_DIR=runs/rhizome_mf2d_w64 slurm/train_rhizome_mf2d.sbatch
#SBATCH --job-name=rhz-mf2d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rhz-mf2d-%j.out
#SBATCH --error=logs/rhz-mf2d-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh
CACHE="${CACHE:-$WORK/data/compressed/rhizome_mf_slices.h5}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
RESUME=()
[ -f "$RUN_DIR/last.pt" ] && RESUME=(--resume)
echo "RUN_DIR: $RUN_DIR"; echo "ARGS: ${ARGS:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# shellcheck disable=SC2086
python -m ebm21cm.train_rhizome_mf2d --cache "$CACHE" --run-dir "$RUN_DIR" --device cuda \
    --threads "${SLURM_CPUS_PER_TASK:-4}" "${RESUME[@]}" ${ARGS:-}
