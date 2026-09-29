#!/usr/bin/env bash

# Launch the frozen seed-43 set of 30 tasks with zero overlap against every existing task set.
#
# Environment prep, resume, and the official batch logic all reuse the tested seed42 entry point; here we only
# pin TASK_SEED, so duplicating the long script does not let the two startup boundaries gradually diverge.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TASK_SEED=43

exec "${SCRIPT_DIR}/run_evaluation_v2_seed42.sh" 30
