#!/bin/bash
# x_HI + T_b cones from the multi-field 2-D slice-wise rhizome (RUN), mirroring a
# multi-field eval_cubes manifest (REFERENCE); NATIVE_CONES optional.
#   sbatch --export=ALL,RUN=runs/rhizome_mf2d_w64_structured,TAG=rhizome2d_mf_w64,REFERENCE=.../manifest.json slurm/export_slicewise_mf.sbatch
#SBATCH --job-name=rhz-slicewise-mf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rhz-slicewise-mf-%j.out
#SBATCH --error=logs/rhz-slicewise-mf-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh
nvidia-smi -L || true
python -m ebm21cm.export_slicewise_mf --run "${RUN:?set RUN}" --reference "${REFERENCE:?set REFERENCE}" \
    --tag "${TAG:?set TAG}" --out-dir "${OUT_DIR:-$WORK/data/eval_cubes}" ${NATIVE_CONES:+--native-cones $NATIVE_CONES} ${NATIVE_ONLY:+--native-only}
