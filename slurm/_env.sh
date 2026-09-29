# Sourced by every sbatch script: project root, conda env, PYTHONPATH.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
module load devel/miniforge >/dev/null 2>&1 || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-fno-env}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
WORK="${WORK_DIR:-/pfs/10/work/hd_id260-fno_training}"
