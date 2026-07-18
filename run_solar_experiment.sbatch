#!/usr/bin/env bash
#SBATCH --job-name=solar-latent
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

CONDA_ENV="${CONDA_ENV:-bp-reservoir}"
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is not available. Load your cluster's Anaconda/Miniconda module first." >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/bp-reservoir-matplotlib-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Project: $PROJECT_DIR"
echo "Conda environment: $CONDA_ENV"
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

srun python -u run_solar_experiment.py \
    --models reservoir scinet \
    --seeds 0 1 2 3 4 \
    --latent-size 2 \
    --nodes-1 150 \
    --nodes-2 150 \
    --encoder-steps 3 \
    --decoder-steps 3 \
    --train-samples 95000 \
    --validation-samples 5000 \
    --test-samples 5000 \
    --series-length 50 \
    --delta-days 7 \
    --sampling-mode independent_catalog \
    --phase-steps 1000 1000 1000 1000 11000 \
    --phase-batch-sizes 256 1024 1024 2048 2048 \
    --phase-learning-rates 1e-4 1e-4 1e-4 1e-5 1e-5 \
    --phase-betas 0.1 0.1 0.1 0.01 0.001 \
    --phase-horizons 20 20 50 50 50 \
    --device auto \
    --output-dir "results/solar-replication-${SLURM_JOB_ID:-local}"
