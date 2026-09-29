#!/bin/bash
# Hyperparameter sweep: array task i runs line i+1 of SWEEP_FILE ("<name> <train_rhizome options>")
# as a budget-limited run in $SWEEP_DIR/<name>. Budget stops never touch the test split,
# so runs are ranked on validation only: python -m ebm21cm.sweep_summary $SWEEP_DIR
#   N=$(grep -cv '^#' sweep.txt)
#   sbatch --array=0-$((N-1)) --export=ALL,SWEEP_FILE=sweep.txt,SWEEP_DIR=runs/sweep slurm/sweep_rhizome.sbatch
#SBATCH --job-name=rhz-sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rhz-sweep-%A_%a.out
#SBATCH --error=logs/rhz-sweep-%A_%a.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh

CACHE="${CACHE:-$WORK/data/compressed/rhizome_xhi_slices.h5}"
SWEEP_FILE="${SWEEP_FILE:?set SWEEP_FILE}"
SWEEP_DIR="${SWEEP_DIR:?set SWEEP_DIR}"
LINE=$(grep -v '^#' "$SWEEP_FILE" | grep -v '^[[:space:]]*$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -n "$LINE" ] || { echo "no line $SLURM_ARRAY_TASK_ID in $SWEEP_FILE" >&2; exit 1; }
read -r NAME OPTIONS <<<"$LINE"
RUN_DIR="$SWEEP_DIR/$NAME"
RESUME=()
[ -f "$RUN_DIR/last.pt" ] && RESUME=(--resume)
echo "task $SLURM_ARRAY_TASK_ID: $NAME $OPTIONS ${COMMON:-}"
# shellcheck disable=SC2086
python -m ebm21cm.train_rhizome --cache "$CACHE" --run-dir "$RUN_DIR" --device cuda --data memory \
    --threads "${SLURM_CPUS_PER_TASK:-4}" "${RESUME[@]}" ${COMMON:-} $OPTIONS
