#!/usr/bin/env bash
#SBATCH --job-name=narma-aggregate
#SBATCH --output=narma/jobs/logs/%x-%j.out
#SBATCH --error=narma/jobs/logs/%x-%j.out
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

source narma/jobs/_environment.sh

srun python -u -m narma.run_benchmark aggregate "${NARMA_COMMON_ARGS[@]}"
