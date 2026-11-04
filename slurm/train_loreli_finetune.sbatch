#!/bin/bash
# One fine-tuning ladder arm on LoReLi II.
#   ARGS="--cache ... --run-dir ... --init-from ... --freeze decoder --train-blocks 8 ..." \
#     sbatch --export=ALL,ARGS="$ARGS" slurm/train_loreli_finetune.sbatch
#SBATCH --job-name=loreli-ft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=logs/loreli-ft-%j.out
#SBATCH --error=logs/loreli-ft-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh
mkdir -p logs
# shellcheck disable=SC2086
python -m ebm21cm.train_loreli_finetune ${ARGS:?set ARGS}
