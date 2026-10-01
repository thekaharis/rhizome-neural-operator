#!/bin/bash
# Export full-cone test predictions of a 3-D rhizome (RHIZOME=run dir) or fno-21cm
# LOS-window model (FNO_CKPT=best.pt) in fno-21cm's eval_cubes formats.
#   sbatch --export=ALL,RHIZOME=runs/rhizome3d/w48_m36_h200,TAG=rhizome3d_w48 slurm/export_cubes3d.sbatch
#SBATCH --job-name=rhz-export3d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rhz-export3d-%j.out
#SBATCH --error=logs/rhz-export3d-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh

TAG="${TAG:?set TAG}"
OUT_DIR="${OUT_DIR:-$WORK/data/eval_cubes}"
H5_CONES="${H5_CONES:-24 27 1031 1275 1625 1684 1892}"
if [ -n "${RHIZOME:-}" ]; then SRC=(--rhizome "$RHIZOME"); else SRC=(--fno "${FNO_CKPT:?set RHIZOME or FNO_CKPT}"); fi
nvidia-smi -L || true
# shellcheck disable=SC2086
python -m ebm21cm.export_cubes3d "${SRC[@]}" --tag "$TAG" --out-dir "$OUT_DIR" --split "${SPLIT:-test}" \
    --h5-cones $H5_CONES
