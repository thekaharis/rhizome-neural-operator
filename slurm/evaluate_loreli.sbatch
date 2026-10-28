#!/bin/bash
# Zero-shot LoReLi II evaluation (ebm21cm.evaluate_loreli) on one A100.
#   ARGS="--run /pfs/.../runs/rhizome_mf2d_w64_structured_all --data /pfs/.../loreli_lc128 \
#         --cache /pfs/.../rhizome_mf_slices_all.h5 \
#         --out /pfs/10/work/hd_id260-fno_training/handoff/rhizome-loreli/runs/eval3d/loreli_lc128/rhizome_mf2d_w64_structured_all" \
#     sbatch --export=ALL,ARGS="$ARGS" slurm/evaluate_loreli.sbatch
#SBATCH --job-name=loreli-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/loreli-eval-%j.out
#SBATCH --error=logs/loreli-eval-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh
mkdir -p logs
# shellcheck disable=SC2086
python -m ebm21cm.evaluate_loreli ${ARGS:?set ARGS}
