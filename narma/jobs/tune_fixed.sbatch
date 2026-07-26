#!/usr/bin/env bash
#SBATCH --job-name=narma-tune-fixed
#SBATCH --output=narma/jobs/logs/%x-%A_%a.out
#SBATCH --error=narma/jobs/logs/%x-%A_%a.out
#SBATCH --array=0-39%20
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

source narma/jobs/_environment.sh

srun python -u -m narma.run_benchmark tune-array \
    "${NARMA_COMMON_ARGS[@]}" \
    --array-models \
        esn_ridge \
        large_esn_ridge \
        deep_esn_ridge \
        random_bottleneck \
        pca_bottleneck \
    --index "${SLURM_ARRAY_TASK_ID:?Slurm array task ID is required}"
