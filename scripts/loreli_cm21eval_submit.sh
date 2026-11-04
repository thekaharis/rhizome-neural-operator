#!/bin/bash
# Chain 21cmEVAL for a group of models: GPU prediction job -> one CPU 21cmEVAL job per model -> merge.
#
#   scripts/loreli_cm21eval_submit.sh "reference reference_oracle b8_decoder" [AFTER_JOB_IDS]
#
# AFTER_JOB_IDS: colon-separated training jobs to wait for (afterany: a failed arm is skipped by the
# prediction step and its 21cmEVAL job fails visibly; the others still run).
# Output: runs/cm21eval/<model>/{xhi,tb}/ and runs/cm21eval/compare.{md,csv} (rebuilt by every merge).
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS="$1"
DEP=${2:-}
OUT=$PWD/runs/cm21eval
mkdir -p "$OUT" logs
dep_arg=()
[ -n "$DEP" ] && dep_arg=(--dependency=afterany:$DEP)
pred=$(MODELS="$MODELS" PRED_ARGS="--skip-missing" sbatch --parsable "${dep_arg[@]}" \
       --job-name="cm21-pred" --export=ALL slurm/loreli_cm21eval_predict.sbatch)
echo "predict $pred: $MODELS ${DEP:+(after $DEP)}"
evals=()
for m in $MODELS; do
    j=$(MODELS="$m" OUT="$OUT/$m" sbatch --parsable --dependency=afterok:$pred --job-name="cm21-$m" \
        --export=ALL slurm/loreli_cm21eval.sbatch)
    evals+=("$j"); echo "  21cmEVAL $j: $m"
done
merge=$(sbatch --parsable --dependency=afterany:$(IFS=:; echo "${evals[*]}") --job-name=cm21-merge \
        --partition=compute --cpus-per-task=1 --mem=4G --time=00:10:00 --output=logs/cm21eval-merge-%j.out \
        --wrap "source slurm/_env.sh && python scripts/loreli_cm21eval_merge.py $OUT")
echo "  merge $merge"
