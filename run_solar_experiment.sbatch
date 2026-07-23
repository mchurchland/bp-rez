#!/usr/bin/env bash
#SBATCH --job-name=solar-grid-15k
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out
#SBATCH --time=10:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

# Some cluster Conda distributions (notably Intel Python) have deactivate
# hooks that read optional variables without defaults. Enable nounset only
# after Conda has finished running those hooks.
set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

CONDA_ENV="${CONDA_ENV:-bp-reservoir}"
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is not available. Load your cluster's Anaconda/Miniconda module first." >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export MPLCONFIGDIR="${TMPDIR:-/tmp}/bp-reservoir-matplotlib-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Project: $PROJECT_DIR"
echo "Conda environment: $CONDA_ENV"
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

GRID_OUTPUT_DIR="${GRID_OUTPUT_DIR:-results/solar-grid-15k-${SLURM_JOB_ID:-local}}"

srun python -u run_solar_grid_search.py \
    --seed 0 \
    --data-seed 2026 \
    --time-budget-hours 10 \
    --device auto \
    --output-dir "$GRID_OUTPUT_DIR"
