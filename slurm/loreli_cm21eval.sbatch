#!/bin/bash
# 21cmEVAL on precomputed LoReLi test-lightcone predictions (CPU): x_HI pass + T_b pass per geometry group.
#
#   MODELS="b8_decoder" OUT=runs/cm21eval/b8_decoder sbatch slurm/loreli_cm21eval.sbatch
#
# MODELS  names with predictions under $PREDS/<model>/ (scripts/loreli_cm21eval_predict.py)
# OUT     output dir; <group>/{xhi,tb}/ hold results.json, summary.csv, stages.csv, report.md, curves.npz
# EVAL_GROUPS  optional subset of the groups under $DATA (default: all)
# XHI_METRICS / TB_METRICS  metric lists of the two passes; EXTRA  further cm21eval arguments
#   (e.g. EXTRA="--set cyl_native_cell_mpc=2.296875 --set cyl_native_los_dz=0.0065146580"
#    with XHI_METRICS=TB_METRICS=cylindrical,gallery; the plots merge such runs with the full ones)
#SBATCH --job-name=cm21eval
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cm21eval-%j.out
#SBATCH --error=logs/cm21eval-%j.err
set -euo pipefail
source slurm/_env.sh
export PYTHONPATH="$HOME/21cmEVAL/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
DATA=${DATA:-/pfs/10/work/hd_id260-fno_training/data/eval_cubes/loreli_l294_v3_test}
export LORELI_EVAL_PREDS=${PREDS:-$PWD/runs/cm21eval/preds}
ADAPTER=$PWD/scripts/loreli_cm21eval_adapter.py
args=()
for m in $MODELS; do args+=(--model "$m=python:$ADAPTER:predict_$m"); done
common=(--split test --set omega_b=0.04877 --set hubble_h=0.678 --quiet)
read -r -a extra <<< "${EXTRA:-}"
common+=("${extra[@]}")
EVAL_GROUPS=${EVAL_GROUPS:-$(cd "$DATA" && ls -d */ | tr -d /)}
echo "models: $MODELS  groups: $EVAL_GROUPS  out: $OUT  commit: $(git rev-parse --short HEAD)  21cmEVAL: $(git -C $HOME/21cmEVAL rev-parse --short HEAD)"
for g in $EVAL_GROUPS; do
    echo "== $g $(date +%T)"
    python scripts/loreli_cm21eval_run.py run --data "$DATA/$g/manifest.json" --target xhi --set target=xhi \
        --out "$OUT/$g/xhi" --title "LoReLi II v3 test, group $g, x_HI" "${common[@]}" "${args[@]}" \
        --metrics ${XHI_METRICS:-voxel,edge,spectrum,bubble,morphology,history,observable,eft,front,parity,anisotropy}
    python scripts/loreli_cm21eval_run.py run --data "$DATA/$g/manifest_tb.json" --target tb --set target=tb \
        --out "$OUT/$g/tb" --title "LoReLi II v3 test, group $g, T_b" "${common[@]}" "${args[@]}" \
        --metrics ${TB_METRICS:-voxel,edge,spectrum}
done
echo "done $(date +%T)"
