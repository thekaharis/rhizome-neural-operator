#!/bin/bash
# Sample an ensemble per held-out slice, then score it.
#   sbatch --export=ALL,RUN_DIR=runs/base slurm/sample_eval.sbatch
#SBATCH --job-name=ebm-sample
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ebm-sample-%j.out
#SBATCH --error=logs/ebm-sample-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh

RUN_DIR="${RUN_DIR:?set RUN_DIR}"
SPLIT="${SPLIT:-test}"
TAG="${TAG:-${SPLIT}_s${STEPS:-32}_m${MALA_STEPS:-0}}"
OUT="$RUN_DIR/samples_$TAG.h5"
python -m ebm21cm.sample --run-dir "$RUN_DIR" --split "$SPLIT" --n-rows "${N_ROWS:-128}" \
    --n-samples "${N_SAMPLES:-16}" --batch-size "${BATCH:-16}" --steps "${STEPS:-32}" \
    --mala-steps "${MALA_STEPS:-0}" --churn "${CHURN:-0}" --device cuda \
    ${CANDIDATES:+--candidates "$CANDIDATES"} --out "$OUT"
python -m ebm21cm.evaluate --samples "$OUT" --out-dir "$RUN_DIR/eval_$TAG" ${Z_EDGES:+--z-edges "$Z_EDGES"}
