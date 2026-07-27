#!/usr/bin/env bash
# Submit the approximately one-hour NARMA screen with up to 20 GPU workers.

set -euo pipefail

export NARMA_PROFILE=light
exec bash narma/jobs/submit_publication_benchmark.sh
