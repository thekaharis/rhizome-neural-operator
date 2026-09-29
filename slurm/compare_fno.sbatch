#!/bin/bash
# Rhizome vs fno-21cm 2-D x_HI predictions on identical held-out slices (viz, A30).
# MODULE=compare_fno (figures + metrics, default) or edge_compare (boundary-band edges).
#   sbatch --export=ALL,RUN_DIR=runs/sweep1/w128 slurm/compare_fno.sbatch
#   sbatch --export=ALL,RUN_DIR=runs/sweep1/w128,MODULE=edge_compare slurm/compare_fno.sbatch
#SBATCH --job-name=rhz-cmp-fno
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rhz-cmp-fno-%j.out
#SBATCH --error=logs/rhz-cmp-fno-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh

RUN_DIR="${RUN_DIR:?set RUN_DIR}"
FNO_ROOT="${FNO_ROOT:-$WORK/fno-21cm}"
CK="$FNO_ROOT/checkpoints/2d_xhi"
MODULE="${MODULE:-compare_fno}"
OUT="${OUT:-runs/$MODULE/$(basename "$RUN_DIR")}"
FNO_RUNS="${FNO_RUNS:-LWF=$CK/fno_lwf/xhi2d_lwf_glob_e100 WHNO=$CK/fno_whno/xhi2d_whno_glob_e100 LocalFNO=$CK/localfno/xhi2d_localfno_lr3e4_e30 UFNO=$CK/ufno/xhi2d_ufno_lr3e4_e30}"
# shellcheck disable=SC2086
python -m "ebm21cm.$MODULE" --rhizome "$RUN_DIR" --fno-root "$FNO_ROOT" --fno $FNO_RUNS --out "$OUT" --device cuda
