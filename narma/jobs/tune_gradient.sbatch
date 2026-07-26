#!/usr/bin/env bash
#SBATCH --job-name=narma-tune-gradient
#SBATCH --output=narma/jobs/logs/%x-%A_%a.out
#SBATCH --error=narma/jobs/logs/%x-%A_%a.out
#SBATCH --array=0-23%20
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

source narma/jobs/_environment.sh

srun python -u -m narma.run_benchmark tune-array \
    "${NARMA_COMMON_ARGS[@]}" \
    --array-models \
        learned_linear \
        learned_nonlinear \
        gru \
    --index "${SLURM_ARRAY_TASK_ID:?Slurm array task ID is required}"
