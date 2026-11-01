#!/bin/bash
# Evaluate LoReLi checkpoints on a cache split (default test).
#   ARGS="--run runs/loreli_finetune/<run> --cache .../loreli_slices_l294_v3.h5 --split test" \
#     sbatch --export=ALL,ARGS="$ARGS" slurm/evaluate_loreli_finetune.sbatch
#SBATCH --job-name=loreli-ev
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=logs/loreli-ev-%j.out
#SBATCH --error=logs/loreli-ev-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh
mkdir -p logs
# shellcheck disable=SC2086
python -m ebm21cm.evaluate_loreli_finetune ${ARGS:?set ARGS}
