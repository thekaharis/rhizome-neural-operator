#!/bin/bash
# Build the LoReLi fine-tuning slice cache with the evaluation adapter.
#   ARGS="--out /pfs/.../runs/loreli_cache/loreli_slices_l294_v2.h5 \
#     --split-file .../block_split_v2_seed0.json --assigned-only ..." \
#     sbatch --export=ALL,ARGS="$ARGS" slurm/build_loreli_cache.sbatch
#SBATCH --job-name=loreli-cache
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=logs/loreli-cache-%j.out
#SBATCH --error=logs/loreli-cache-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh
mkdir -p logs
# shellcheck disable=SC2086
python -m ebm21cm.data.loreli_cache build ${ARGS:?set ARGS}
