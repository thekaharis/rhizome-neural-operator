#!/bin/bash
# Merge the shards of slurm/build_cache.sbatch and compute train-only statistics.
#SBATCH --job-name=ebm-merge
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ebm-merge-%j.out
#SBATCH --error=logs/ebm-merge-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh

SHARD_DIR="${SHARD_DIR:-$WORK/data/compressed/ebm_slices_shards}"
CACHE="${CACHE:-$WORK/data/compressed/ebm_slices.h5}"
[ -e "$CACHE" ] && { echo "$CACHE exists; refusing to overwrite" >&2; exit 1; }
python -m ebm21cm.data.cache merge --out "$CACHE" "$SHARD_DIR"/shard*.h5
