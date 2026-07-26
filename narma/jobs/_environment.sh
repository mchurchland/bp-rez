#!/usr/bin/env bash
# Shared environment setup. This file is sourced by the NARMA Slurm workers.

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
if [[ ! -d "$PROJECT_DIR/narma" ]]; then
    echo "Submit the job from the bp-reservoir repository root." >&2
    exit 1
fi

CONDA_ENV="${CONDA_ENV:-bp-reservoir}"
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is unavailable. Load Anaconda/Miniconda before submitting." >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bp-narma-mpl-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

NARMA_PROFILE="${NARMA_PROFILE:-screening}"
case "$NARMA_PROFILE" in
    screening)
        PROFILE_OUTPUT_ROOT="narma/results/screening_benchmark"
        PROFILE_SEARCH_TRIALS=4
        PROFILE_MAX_EPOCHS=600
        PROFILE_MAX_EPOCH_CAP_FRACTION=1
        PROFILE_FINAL_PAIR_IDS="0 1 2"
        PROFILE_TUNING_PAIR_IDS="10000 10001"
        ;;
    publication)
        PROFILE_OUTPUT_ROOT="narma/results/publication_benchmark"
        PROFILE_SEARCH_TRIALS=8
        PROFILE_MAX_EPOCHS=1000
        PROFILE_MAX_EPOCH_CAP_FRACTION=0.1
        PROFILE_FINAL_PAIR_IDS="0 1 2 3 4 5 6 7 8 9"
        PROFILE_TUNING_PAIR_IDS="10000 10001 10002"
        ;;
    *)
        echo "NARMA_PROFILE must be screening or publication." >&2
        exit 1
        ;;
esac

NARMA_OUTPUT_ROOT="${NARMA_OUTPUT_ROOT:-$PROFILE_OUTPUT_ROOT}"
NARMA_SEARCH_TRIALS="${NARMA_SEARCH_TRIALS:-$PROFILE_SEARCH_TRIALS}"
NARMA_MAX_EPOCHS="${NARMA_MAX_EPOCHS:-$PROFILE_MAX_EPOCHS}"
NARMA_MAX_EPOCH_CAP_FRACTION="${NARMA_MAX_EPOCH_CAP_FRACTION:-$PROFILE_MAX_EPOCH_CAP_FRACTION}"
NARMA_FINAL_PAIR_IDS="${NARMA_FINAL_PAIR_IDS:-$PROFILE_FINAL_PAIR_IDS}"
NARMA_TUNING_PAIR_IDS="${NARMA_TUNING_PAIR_IDS:-$PROFILE_TUNING_PAIR_IDS}"
read -r -a NARMA_FINAL_PAIR_ID_ARGS <<< "$NARMA_FINAL_PAIR_IDS"
read -r -a NARMA_TUNING_PAIR_ID_ARGS <<< "$NARMA_TUNING_PAIR_IDS"

NARMA_COMMON_ARGS=(
    --output-root "$NARMA_OUTPUT_ROOT"
    --orders 5 10 20 30
    --regimes small long
    --models
        esn_ridge
        large_esn_ridge
        deep_esn_ridge
        random_bottleneck
        pca_bottleneck
        learned_linear
        learned_nonlinear
        gru
    --final-pair-ids "${NARMA_FINAL_PAIR_ID_ARGS[@]}"
    --tuning-pair-ids "${NARMA_TUNING_PAIR_ID_ARGS[@]}"
    --small-train-length 2000
    --long-train-length 10000
    --validation-length 2000
    --test-length 10000
    --washout 200
    --search-trials "$NARMA_SEARCH_TRIALS"
    --max-epochs "$NARMA_MAX_EPOCHS"
    --max-epoch-cap-fraction "$NARMA_MAX_EPOCH_CAP_FRACTION"
    --early-stopping-patience 100
    --scheduler-patience 20
    --scheduler-factor 0.5
    --minimum-learning-rate 1e-6
    --gradient-clip 1.0
    --nodes-1 150
    --nodes-2 150
    --latent-size 10
    --gru-hidden-size 22
    --device auto
)

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Host: $(hostname)"
echo "Project: $PROJECT_DIR"
echo "Output root: $NARMA_OUTPUT_ROOT"
echo "Benchmark profile: $NARMA_PROFILE"
echo "Conda environment: $CONDA_ENV"
echo "Python: $(command -v python)"
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
