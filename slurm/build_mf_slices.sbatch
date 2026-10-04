#!/bin/bash
# Array job: one multi-field slice-cache shard per task (ebm21cm.data.mf_slices),
# read through fno-21cm's multi-field preparation (PREPARATION=... to override;
# CLEAN_TB=1 rebuilds T_b from raw files). Then merge:
#   AID=$(sbatch --parsable slurm/build_mf_slices.sbatch)
#   sbatch --dependency=afterok:$AID --export=ALL,MODE=merge slurm/build_mf_slices.sbatch   (as a non-array job)
#SBATCH --job-name=mf-slices
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mf-slices-%A_%a.out
#SBATCH --error=logs/mf-slices-%A_%a.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
SHARD_DIR="${SHARD_DIR:-$WORK/data/compressed/rhizome_mf_slices_shards}"
CACHE="${CACHE:-$WORK/data/compressed/rhizome_mf_slices.h5}"
mkdir -p "$SHARD_DIR"
if [ "${MODE:-build}" = merge ]; then
    [ -e "$CACHE" ] && { echo "$CACHE exists; refusing to overwrite" >&2; exit 1; }
    python -m ebm21cm.data.mf_slices merge --out "$CACHE" "$SHARD_DIR"/shard*.h5
else
    N="${SLURM_ARRAY_TASK_COUNT:?submit as an array job}"; S="${SLURM_ARRAY_TASK_ID:?}"
    python -m ebm21cm.data.mf_slices build --out "$SHARD_DIR/shard$(printf %03d "$S").h5" --shard "$S" --n-shards "$N" \
        --slices-per-cone "${SLICES_PER_CONE:-48}" --background-slices "${BACKGROUND_SLICES:-6}" \
        ${PREPARATION:+--preparation "$PREPARATION"} ${CLEAN_TB:+--clean-tb}
fi
echo "done $(date)"
