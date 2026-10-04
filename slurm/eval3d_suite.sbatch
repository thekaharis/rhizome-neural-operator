#!/bin/bash
# fno-21cm's 3-D evaluation suite on exported cubes of several models (CPU only).
# MANIFESTS: space-separated eval_cubes manifests; the first model is the reference.
# The suite needs identical ordered cones and truth across models, so the manifests
# are merged on their shared cones first (truth equality is checked).
#   sbatch --export=ALL,NAME=rhizome_vs_fno,MANIFESTS="a/manifest.json b/manifest.json" slurm/eval3d_suite.sbatch
# DTB=1 adds the model-own dT_b physical evaluation (multi-field exports only); ONLY_DTB=1 runs just that.
#SBATCH --job-name=rhz-eval3d
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rhz-eval3d-%j.out
#SBATCH --error=logs/rhz-eval3d-%j.err
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" 2>/dev/null || source slurm/_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

NAME="${NAME:?set NAME}"
MANIFESTS="${MANIFESTS:?set MANIFESTS}"
FNO_ROOT="${FNO_ROOT:-$WORK/fno-21cm}"
OUT="$(pwd)/runs/eval3d/$NAME"
mkdir -p "$OUT"
# shellcheck disable=SC2086
python - "$OUT/manifest.json" $MANIFESTS <<'PY'
import json, sys
import numpy as np
out, paths = sys.argv[1], sys.argv[2:]
specs = [json.load(open(p)) for p in paths]
grids = [np.asarray(s["z_grid"]) for s in specs]
assert all(np.allclose(g, grids[0]) for g in grids), "manifests use different z grids"
models = {}
for s in specs:
    for name, entries in s["models"].items():
        assert name not in models, f"duplicate model {name}"
        models[name] = {e["cone_id"]: e for e in entries}
shared = sorted(set.intersection(*(set(m) for m in models.values())))
ref = next(iter(models))
for name, by_id in models.items():
    for c in shared:
        with np.load(models[ref][c]["npz"]) as a, np.load(by_id[c]["npz"]) as b:
            assert np.array_equal(a["truth"], b["truth"]), f"truth differs for {name} cone {c}"
merged = {"z_grid": grids[0].tolist(), "models": {n: [m[c] for c in shared] for n, m in models.items()},
          "sources": paths, "shared_cones": len(shared)}
json.dump(merged, open(out, "w"), indent=1)
print(f"merged {len(models)} models on {len(shared)} shared cones -> {out}")
PY
REF=$(python -c "import json;print(next(iter(json.load(open('$OUT/manifest.json'))['models'])))")
cd "$FNO_ROOT"
run() { echo "=== $* ($(date +%T))"; "$@" || echo "!!! failed: $1 $2"; }
# ONLY_DTB=1: just the model-own dT_b evaluation (e.g. T_b-only variants of one x_HI model).
[ "${ONLY_DTB:-0}" = 1 ] && DTB=1
[ "${ONLY_DTB:-0}" = 1 ] || run python -m viz.physical_evaluation --manifest "$OUT/manifest.json" --out "$OUT/physical"
# DTB=1 (all models export tb_pred): also score each model's own dT_b, with a
# pre-reionization stage that covers the absorption trough.
if [ "${DTB:-0}" = 1 ]; then
    run python -m viz.physical_evaluation --manifest "$OUT/manifest.json" --out "$OUT/physical_dtb" \
        --dtb-from-npz --stage-edges 0.02 0.2 0.4 0.6 0.8 0.98 1.001
fi
if [ "${ONLY_DTB:-0}" = 1 ]; then echo "eval3d suite done (ONLY_DTB): $OUT ($(date))"; exit 0; fi
run python -m viz.bubble_size_evaluation --manifest "$OUT/manifest.json" --out "$OUT/bubble_size"
run python -m viz.power_spectrum_evaluation --manifest "$OUT/manifest.json" --out "$OUT/power_spectrum"
run python -m viz.boundary_band_diagnostic --manifest "$OUT/manifest.json" --mode slice --reference "$REF" --out "$OUT/edge_slice"
run python -m viz.boundary_band_diagnostic --manifest "$OUT/manifest.json" --mode 3d --reference "$REF" --out "$OUT/edge_3d"
# Native-resolution per-cone pages for every model that exported them.
python -c "import json;[print(json.load(open(p))['models'].__iter__().__next__(), p) for p in '$MANIFESTS'.split()]" | while read -r tag path; do
    native="$(dirname "$path")/native"
    ls "$native"/*.h5 >/dev/null 2>&1 || continue
    run python -m viz.multifield_detailed --prediction "$native"/*.h5 --out-dir "$OUT/detailed/$tag" --note "$tag"
    run python -m viz.multifield_lightcone_strips --prediction "$native"/*.h5 --out-dir "$OUT/strips/$tag" --crop --note "$tag"
    run python -m viz.multifield_slices --prediction "$native"/*.h5 --n-slices 5 --out-dir "$OUT/slices/$tag"
done
echo "eval3d suite done: $OUT ($(date))"
