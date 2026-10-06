#!/bin/bash
# fno-21cm's 3-D evaluation suite on exported cubes of several models (CPU only),
# run by ebm21cm.suite_cached: steps in parallel, per-model results cached next to
# each model's cubes, so only new models are computed.
# MANIFESTS: space-separated eval_cubes manifests; the first model is the reference.
# The suite needs identical ordered cones and truth across models, so the manifests
# are merged on their shared cones first (truth equality is checked).
#   sbatch --export=ALL,NAME=rhizome_vs_fno,MANIFESTS="a/manifest.json b/manifest.json" slurm/eval3d_suite.sbatch
# DTB=1 adds the model-own dT_b physical evaluation (multi-field exports only); ONLY_DTB=1 runs just that.
#SBATCH --job-name=rhz-eval3d
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rhz-eval3d-%j.out
#SBATCH --error=logs/rhz-eval3d-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

NAME="${NAME:?set NAME}"
MANIFESTS="${MANIFESTS:?set MANIFESTS}"
FLAGS=()
[ "${DTB:-0}" = 1 ] && FLAGS+=(--dtb)
[ "${ONLY_DTB:-0}" = 1 ] && FLAGS+=(--only-dtb)
# shellcheck disable=SC2086
python -m ebm21cm.suite_cached --name "$NAME" --manifests $MANIFESTS "${FLAGS[@]}" \
    --jobs "${JOBS:-6}" --threads "${THREADS:-2}"
