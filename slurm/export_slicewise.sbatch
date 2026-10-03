#!/bin/bash
# Slice-wise 3-D cones from a 2-D rhizome (RUN=train_rhizome run dir), mirroring a
# reference eval_cubes manifest (REFERENCE), restricted to 2-D cache SPLIT cones.
#   sbatch --export=ALL,RUN=runs/rhizome_xhi_w128,TAG=rhizome2d_w128_slicewise,REFERENCE=.../manifest.json slurm/export_slicewise.sbatch
#SBATCH --job-name=rhz-slicewise
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rhz-slicewise-%j.out
#SBATCH --error=logs/rhz-slicewise-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh
nvidia-smi -L || true
python -m ebm21cm.export_slicewise --run "${RUN:?set RUN}" --reference "${REFERENCE:?set REFERENCE}" \
    --tag "${TAG:?set TAG}" --out-dir "${OUT_DIR:-$WORK/data/eval_cubes}" --split "${SPLIT:-eval}" \
    ${NATIVE_CONES:+--native-cones $NATIVE_CONES} ${NATIVE_ONLY:+--native-only}
