#!/bin/bash
# 21cmEVAL on 21cmFAST test cones from eval_cubes exports (CPU): x_HI pass + T_b pass for one chunk per array task.
#
#   python scripts/eval_cubes_cm21eval.py --cubes $CUBES/rhizome2d_mf_w64_all --out $CUBES/21cmfast_test_eval21
#   MODELS="rhizome=rhizome2d_mf_w64_all ufno=mf_cnn_whno_es_tbs_histtb_tbclean_ep26" OUT=runs/cm21eval_21cmfast/main \
#       sbatch --array=0-7 --export=ALL slurm/cm21eval_eval_cubes.sbatch
#
# MODELS  name=eval_cubes-tag pairs (scripts/eval_cubes_adapter.py reads $EVAL_CUBES/<tag>/)
# DATA    chunk manifests (default $CUBES/21cmfast_test_eval21); task i runs chunk C<i>
# XHI_METRICS / TB_METRICS / EXTRA as in slurm/loreli_cm21eval.sbatch
#SBATCH --job-name=cm21eval-21cmfast
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/cm21eval-21cmfast-%A_%a.out
#SBATCH --error=logs/cm21eval-21cmfast-%A_%a.err
set -euo pipefail
source slurm/_env.sh
export PYTHONPATH="$HOME/21cmEVAL/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export EVAL_CUBES=${EVAL_CUBES:-/pfs/10/work/hd_id260-fno_training/data/eval_cubes}
DATA=${DATA:-$EVAL_CUBES/21cmfast_test_eval21}
G=$(printf "C%02d" "${SLURM_ARRAY_TASK_ID:?submit as an array job}")
ADAPTER=$PWD/scripts/eval_cubes_adapter.py
args=()
for pair in $MODELS; do args+=(--model "${pair%%=*}=python:$ADAPTER:predict_${pair#*=}"); done
common=(--split test --set stage_field=truth --quiet)
read -r -a extra <<< "${EXTRA:-}"
common+=("${extra[@]}")
echo "models: $MODELS  group: $G  out: $OUT  commit: $(git rev-parse --short HEAD)  21cmEVAL: $(git -C $HOME/21cmEVAL rev-parse --short HEAD)"
python -m eval21 run --data "$DATA/$G/manifest.json" --target xhi --out "$OUT/$G/xhi" \
    --title "21cmFAST test, chunk $G, x_HI" "${common[@]}" "${args[@]}" \
    --metrics ${XHI_METRICS:-voxel,edge,spectrum,bubble,morphology,history,observable,eft,front,parity,anisotropy}
python -m eval21 run --data "$DATA/$G/manifest_tb.json" --target tb --out "$OUT/$G/tb" \
    --title "21cmFAST test, chunk $G, T_b" "${common[@]}" "${args[@]}" \
    --metrics ${TB_METRICS:-voxel,edge,spectrum}
echo "done $(date +%T)"
