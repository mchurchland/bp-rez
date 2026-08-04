#!/usr/bin/env bash
#SBATCH --job-name=solar-2x150-fix
#SBATCH --output=solar/jobs/logs/%x-%j.out
#SBATCH --error=solar/jobs/logs/%x-%j.out
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Two 150-neuron reservoirs, one explicit 2D bottleneck, and 15,000 updates.
# Scale, beta, and Mars-loss weights match the previous 2x150 grid winner so
# the stateless biased decoder, direct latent skip, and ridge head are isolated.
# Submit with:
#   sbatch solar/jobs/stateless_2x150_15k.sbatch
#
# Optional overrides:
#   CONDA_ENV=base sbatch solar/jobs/stateless_2x150_15k.sbatch
#   sbatch --export=ALL,SOLAR_OUTPUT_DIR=solar/results/my-2x150-run \
#     solar/jobs/stateless_2x150_15k.sbatch

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

SOLAR_OUTPUT_DIR="${SOLAR_OUTPUT_DIR:-solar/results/solar-2x150-decoder-fix-15k-${SLURM_JOB_ID:-local}}"
echo "Output: $SOLAR_OUTPUT_DIR"

srun python -u -m solar.run_experiment \
    --models reservoir \
    --seeds 0 \
    --data-seed 2026 \
    --train-samples 95000 \
    --validation-samples 5000 \
    --test-samples 5000 \
    --series-length 50 \
    --delta-days 7 \
    --lifetime-days 25657 \
    --sampling-mode independent_catalog \
    --nodes-1 150 \
    --nodes-2 150 \
    --reservoir-layers 2 \
    --latent-size 2 \
    --encoder-steps 3 \
    --second-reservoir-steps 3 \
    --decoder-bias-scale 1 \
    --spectral-radius 0.9 \
    --density 0.1 \
    --leak-rate 1 \
    --input-scale 0.5 \
    --interlayer-scale 4 \
    --phase-steps 1000 1000 1000 1000 11000 \
    --phase-batch-sizes 256 1024 1024 2048 2048 \
    --phase-learning-rates 1e-4 1e-4 1e-4 1e-5 1e-5 \
    --phase-betas 0 0 0 0 0 \
    --phase-horizons 20 20 50 50 50 \
    --mars-velocity-loss-weight 30 \
    --mars-curvature-loss-weight 10 \
    --readout-ridge-alphas 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 \
    --training-log-interval 100 \
    --validation-interval 250 \
    --validation-subset 1024 \
    --evaluation-batch-size 1024 \
    --gradient-clip-value 10 \
    --analysis-grid-size 35 \
    --device auto \
    --output-dir "$SOLAR_OUTPUT_DIR"
