#!/usr/bin/env bash
# Submit tuning, lock, final and aggregation jobs with strict dependencies.

set -euo pipefail

if [[ ! -d narma/jobs ]]; then
    echo "Run this command from the bp-reservoir repository root." >&2
    exit 1
fi

NARMA_PROFILE="${NARMA_PROFILE:-screening}"
case "$NARMA_PROFILE" in
    light)
        PROFILE_MAX_CONCURRENT=20
        PROFILE_COMPUTE_TIME_LIMIT="01:00:00"
        PROFILE_FINAL_PAIR_IDS="0"
        PROFILE_TUNING_PAIR_IDS="10000"
        ;;
    screening)
        PROFILE_MAX_CONCURRENT=20
        PROFILE_COMPUTE_TIME_LIMIT="24:00:00"
        PROFILE_FINAL_PAIR_IDS="0 1 2"
        PROFILE_TUNING_PAIR_IDS="10000 10001"
        ;;
    publication)
        PROFILE_MAX_CONCURRENT=8
        PROFILE_COMPUTE_TIME_LIMIT="24:00:00"
        PROFILE_FINAL_PAIR_IDS="0 1 2 3 4 5 6 7 8 9"
        PROFILE_TUNING_PAIR_IDS="10000 10001 10002"
        ;;
    *)
        echo "NARMA_PROFILE must be light, screening or publication." >&2
        exit 1
        ;;
esac
export NARMA_PROFILE

MAX_CONCURRENT="${NARMA_MAX_CONCURRENT:-$PROFILE_MAX_CONCURRENT}"
COMPUTE_TIME_LIMIT="${NARMA_COMPUTE_TIME_LIMIT:-$PROFILE_COMPUTE_TIME_LIMIT}"
if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "NARMA_MAX_CONCURRENT must be a positive integer." >&2
    exit 1
fi
FINAL_PAIR_IDS_TEXT="${NARMA_FINAL_PAIR_IDS:-$PROFILE_FINAL_PAIR_IDS}"
TUNING_PAIR_IDS_TEXT="${NARMA_TUNING_PAIR_IDS:-$PROFILE_TUNING_PAIR_IDS}"
read -r -a FINAL_PAIR_IDS <<< "$FINAL_PAIR_IDS_TEXT"
read -r -a TUNING_PAIR_IDS <<< "$TUNING_PAIR_IDS_TEXT"
if [[ "${#FINAL_PAIR_IDS[@]}" -lt 1 ]]; then
    echo "NARMA_FINAL_PAIR_IDS must contain at least one integer." >&2
    exit 1
fi
if [[ "${#TUNING_PAIR_IDS[@]}" -lt 1 ]]; then
    echo "NARMA_TUNING_PAIR_IDS must contain at least one integer." >&2
    exit 1
fi
declare -A SEEN_FINAL_PAIR_IDS=()
declare -A SEEN_TUNING_PAIR_IDS=()
for pair_id in "${FINAL_PAIR_IDS[@]}"; do
    if ! [[ "$pair_id" =~ ^[0-9]+$ ]]; then
        echo "NARMA_FINAL_PAIR_IDS must contain nonnegative integers." >&2
        exit 1
    fi
    if [[ -n "${SEEN_FINAL_PAIR_IDS[$pair_id]:-}" ]]; then
        echo "NARMA_FINAL_PAIR_IDS must not contain duplicates." >&2
        exit 1
    fi
    SEEN_FINAL_PAIR_IDS[$pair_id]=1
done
for pair_id in "${TUNING_PAIR_IDS[@]}"; do
    if ! [[ "$pair_id" =~ ^[0-9]+$ ]]; then
        echo "NARMA_TUNING_PAIR_IDS must contain nonnegative integers." >&2
        exit 1
    fi
    if [[ -n "${SEEN_TUNING_PAIR_IDS[$pair_id]:-}" ]]; then
        echo "NARMA_TUNING_PAIR_IDS must not contain duplicates." >&2
        exit 1
    fi
    if [[ -n "${SEEN_FINAL_PAIR_IDS[$pair_id]:-}" ]]; then
        echo "Tuning and final pair IDs must be disjoint." >&2
        exit 1
    fi
    SEEN_TUNING_PAIR_IDS[$pair_id]=1
done
final_task_count=$((4 * 2 * ${#FINAL_PAIR_IDS[@]}))
final_last_index=$((final_task_count - 1))

fixed_submission="$(
    sbatch --parsable --time="$COMPUTE_TIME_LIMIT" \
        --array="0-39%${MAX_CONCURRENT}" \
        narma/jobs/tune_fixed.sbatch
)"
fixed_job="${fixed_submission%%;*}"
gradient_submission="$(
    sbatch --parsable --time="$COMPUTE_TIME_LIMIT" \
        --array="0-23%${MAX_CONCURRENT}" \
        narma/jobs/tune_gradient.sbatch
)"
gradient_job="${gradient_submission%%;*}"
lock_submission="$(
    sbatch --parsable \
        --dependency="afterok:${fixed_job}:${gradient_job}" \
        narma/jobs/lock.sbatch
)"
lock_job="${lock_submission%%;*}"
final_submission="$(
    sbatch --parsable --dependency="afterok:${lock_job}" \
        --time="$COMPUTE_TIME_LIMIT" \
        --array="0-${final_last_index}%${MAX_CONCURRENT}" \
        narma/jobs/final.sbatch
)"
final_job="${final_submission%%;*}"
aggregate_submission="$(
    sbatch --parsable --dependency="afterok:${final_job}" \
        narma/jobs/aggregate.sbatch
)"
aggregate_job="${aggregate_submission%%;*}"

printf 'Fixed-model tuning: %s\n' "$fixed_job"
printf 'Learned tuning:     %s\n' "$gradient_job"
printf 'Configuration lock: %s\n' "$lock_job"
printf 'Final paired array: %s\n' "$final_job"
printf 'Benchmark profile:  %s\n' "$NARMA_PROFILE"
printf 'Tuning pair IDs:     %s\n' "$TUNING_PAIR_IDS_TEXT"
printf 'Final pair IDs:      %s\n' "$FINAL_PAIR_IDS_TEXT"
printf 'Compute time limit:  %s\n' "$COMPUTE_TIME_LIMIT"
printf 'Aggregation:        %s\n' "$aggregate_job"
