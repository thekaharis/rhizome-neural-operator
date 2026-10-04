#!/bin/bash
# LOS smoothing of the multi-field 2-D model's spin variable (ebm21cm.smooth_spin_test).
#   SIGMAS="0 2 4 8 16" sbatch --export=ALL,RUN=...,TAG=...,REFERENCE=... slurm/smooth_spin_test.sbatch
#SBATCH --job-name=rhz-smooth-u
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rhz-smooth-u-%j.out
#SBATCH --error=logs/rhz-smooth-u-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh
nvidia-smi -L || true
# shellcheck disable=SC2086
python -m ebm21cm.smooth_spin_test --run "${RUN:?}" --reference "${REFERENCE:?}" --tag "${TAG:?}" \
    --out-dir "${OUT_DIR:-$WORK/data/eval_cubes}" --sigmas ${SIGMAS:-0 2 4 8 16} ${MAX_CONES:+--max-cones $MAX_CONES}
