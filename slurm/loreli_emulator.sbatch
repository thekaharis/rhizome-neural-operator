#!/bin/bash
# LoReLi history emulator pipeline (CPU).  MODULE selects the step:
#   MODULE=ebm21cm.data.loreli_globals ARGS="build --out .../data/compressed/loreli_globals_v1.h5"
#   MODULE=ebm21cm.loreli_history      ARGS="train --globals ... --split-file ... --out runs/loreli_emulator/v1"
#   MODULE=ebm21cm.loreli_history      ARGS="scalars --globals ... --emulator ... --cache ... --out ..."
#     sbatch --export=ALL,MODULE=...,ARGS="$ARGS" slurm/loreli_emulator.sbatch
#SBATCH --job-name=loreli-emu
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/loreli-emu-%j.out
#SBATCH --error=logs/loreli-emu-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh
mkdir -p logs
# shellcheck disable=SC2086
python -m "${MODULE:?set MODULE}" ${ARGS:?set ARGS}
