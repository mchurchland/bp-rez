#!/usr/bin/env bash
#SBATCH --job-name=solar-2x-size
#SBATCH --output=solar/jobs/logs/%x-%j.out
#SBATCH --error=solar/jobs/logs/%x-%j.out
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

# Submit with:
#   sbatch solar/jobs/reservoir_size_sweep.sbatch
#
# Optional overrides:
#   CONDA_ENV=base sbatch solar/jobs/reservoir_size_sweep.sbatch
#   sbatch --export=ALL,SIZE_SWEEP_OUTPUT_DIR=solar/results/my-size-sweep \
#     solar/jobs/reservoir_size_sweep.sbatch
#   sbatch --export=ALL,RESERVOIR_SIZES="10 50 150 300",MODEL_SEEDS="0" \
#     solar/jobs/reservoir_size_sweep.sbatch

# Some cluster Conda distributions have deactivate hooks that read optional
# variables without defaults. Enable nounset only after Conda activation.
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
echo "Python: $(command -v python)"
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

SIZE_SWEEP_OUTPUT_DIR="${SIZE_SWEEP_OUTPUT_DIR:-solar/results/solar-2x-size-sweep-15k-${SLURM_JOB_ID:-local}}"
read -r -a SIZE_VALUES <<< "${RESERVOIR_SIZES:-10 20 30 50 75 100 150 200 250 300}"
read -r -a SEED_VALUES <<< "${MODEL_SEEDS:-0 1 2}"
echo "Output: $SIZE_SWEEP_OUTPUT_DIR"
echo "Reservoir sizes: ${SIZE_VALUES[*]}"
echo "Model seeds: ${SEED_VALUES[*]}"

srun python -u -m solar.run_size_sweep \
    --sizes "${SIZE_VALUES[@]}" \
    --seeds "${SEED_VALUES[@]}" \
    --data-seed 2026 \
    --device auto \
    --output-dir "$SIZE_SWEEP_OUTPUT_DIR"
