#!/usr/bin/env bash
#SBATCH --job-name=narma-lock
#SBATCH --output=narma/jobs/logs/%x-%j.out
#SBATCH --error=narma/jobs/logs/%x-%j.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

source narma/jobs/_environment.sh

srun python -u -m narma.run_benchmark lock "${NARMA_COMMON_ARGS[@]}"
