#!/bin/bash
# Predict full LoReLi test lightcones (common z grid) for MODELS on one GPU.
#
#   MODELS="reference reference_oracle b8_decoder" sbatch slurm/loreli_cm21eval_predict.sbatch
#SBATCH --job-name=cm21eval-pred
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/cm21eval-pred-%j.out
#SBATCH --error=logs/cm21eval-pred-%j.err
set -euo pipefail
source slurm/_env.sh
echo "models: $MODELS  commit: $(git rev-parse --short HEAD)"
python scripts/loreli_cm21eval_predict.py --models $MODELS ${PRED_ARGS:-}
