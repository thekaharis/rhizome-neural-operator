#!/bin/bash
# CPU-only monotone-output-map diagnostic for the LoReLi zero-shot predictions (HAR-63).
# No GPU, no training; reads the saved predictions and the raw LoReLi cubes.
#SBATCH --job-name=loreli-monotone
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/loreli-monotone-%j.out
#SBATCH --error=logs/loreli-monotone-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh
# scikit-learn is not installed in fno-env; a no-deps copy lives outside the repo at handoff/pylibs.
export PYTHONPATH="/pfs/10/work/hd_id260-fno_training/handoff/pylibs:${PYTHONPATH:-}"
mkdir -p logs
python scripts/loreli_monotone_map.py \
  --pred runs/eval3d/loreli_lc128/rhizome_mf2d_w64_structured_all_full_v2 \
  --data /pfs/10/work/hd_id260-fno_training/data/loreli_lc128 \
  --split runs/loreli_splits/block_split_v2_seed0.json \
  --out runs/eval3d/loreli_lc128/monotone_map_v2
