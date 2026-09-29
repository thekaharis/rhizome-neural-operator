#!/bin/bash
# =============================================================================
# Array job: build one slice-cache shard per task from the raw lightcones.
# CPU only, I/O bound. Every task sees the full file list, so the cone-level
# train/val/test split is identical across shards. Then merge:
#
#     mkdir -p logs
#     AID=$(sbatch --parsable slurm/build_cache.sbatch)
#     sbatch --dependency=afterok:"$AID" slurm/merge_cache.sbatch
#
# The array width sets N_SHARDS automatically. Cones with a non-finite value in
# a sampled slice (33 sims have non-finite brightness_temp voxels) are skipped
# and listed in the shard's `skipped` attribute rather than failing the task.
# =============================================================================
#SBATCH --job-name=ebm-cache
#SBATCH --partition=compute
#SBATCH --array=0-32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ebm-cache-%A_%a.out
#SBATCH --error=logs/ebm-cache-%A_%a.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh

LIGHTCONES="${LIGHTCONES:-$WORK/data/data}"
PATTERN="${PATTERN:-21cmfast_11d_sample*.h5}"
SHARD_DIR="${SHARD_DIR:-$WORK/data/compressed/ebm_slices_shards}"
SLICES_PER_CONE="${SLICES_PER_CONE:-8}"
BANDS="${BANDS:--32:-16,-16:-8,-8:-4,-3,-2,-1,0,1,2,3,4:8,8:16,16:32}"
Z_MIN="${Z_MIN:-5.001}"
Z_MAX="${Z_MAX:-24.97}"
N_SHARDS="${SLURM_ARRAY_TASK_COUNT:?submit as an array job}"
SHARD="${SLURM_ARRAY_TASK_ID:?submit as an array job}"

[ -d "$LIGHTCONES" ] || { echo "LIGHTCONES=$LIGHTCONES does not exist" >&2; exit 1; }
mkdir -p "$SHARD_DIR"
OUT="$SHARD_DIR/shard$(printf %03d "$SHARD").h5"
echo "shard $SHARD/$N_SHARDS  $LIGHTCONES/$PATTERN -> $OUT  ($(date))"
python -m ebm21cm.data.cache build --data "$LIGHTCONES" --pattern "$PATTERN" --out "$OUT" \
    --slices-per-cone "$SLICES_PER_CONE" --bands="$BANDS" --z-min "$Z_MIN" --z-max "$Z_MAX" \
    --shard "$SHARD" --n-shards "$N_SHARDS"
echo "done $(date)"
