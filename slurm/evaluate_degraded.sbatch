#!/bin/bash
# Resolution-only LoReLi transfer: 21cmFAST test cones degraded to LoReLi's grid.
#   ARGS="--run /pfs/.../runs/rhizome_mf2d_w64_structured_all --cache /pfs/.../rhizome_mf_slices_all.h5 \
#         --mirror /pfs/.../native_mirror_4f --out /pfs/.../runs/eval3d/loreli_lc128/degraded" \
#     sbatch --export=ALL,ARGS="$ARGS" slurm/evaluate_degraded.sbatch
#SBATCH --job-name=loreli-deg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=logs/loreli-deg-%j.out
#SBATCH --error=logs/loreli-deg-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh
mkdir -p logs
# shellcheck disable=SC2086
python -m ebm21cm.evaluate_degraded ${ARGS:?set ARGS}
