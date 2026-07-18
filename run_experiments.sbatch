#!/usr/bin/env bash
#SBATCH --job-name=narma10-reservoir
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Some cluster Conda distributions (notably Intel Python) have deactivate
# hooks that read optional variables without defaults. Enable nounset only
# after Conda has finished running those hooks.
set -eo pipefail

# Slurm normally sets this to the directory from which sbatch was called.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# Activate Conda in a non-interactive Slurm shell. Override the default with:
# sbatch --export=ALL,CONDA_ENV=my-environment run_experiments.sbatch
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

srun python -u run_experiments.py \
    --seeds 0 1 2 3 4 \
    --models single_esn deep_esn proposed_nonlinear proposed_linear large_esn \
    --nodes-1 150 \
    --nodes-2 150 \
    --latent-size 10 \
    --train-length 2000 \
    --val-length 500 \
    --test-length 500 \
    --washout 100 \
    --spectral-radius 0.9 \
    --density 0.1 \
    --leak-rate 1.0 \
    --input-scale 0.5 \
    --interlayer-scale 1.0 \
    --learning-rate 0.001 \
    --max-epochs 200 \
    --patience 25 \
    --grad-clip 1.0 \
    --data-seed 2026 \
    --device auto \
    --output-dir "results/narma10-300nodes-${SLURM_JOB_ID:-local}"
