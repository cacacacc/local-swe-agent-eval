#!/usr/bin/env bash

# One-shot launcher for evaluation v2's frozen task set; seed 42 by default, and the seed43 wrapper overrides the seed.
#
# Usage:
#   ./scripts/run_evaluation_v2_seed42.sh 15
#   ./scripts/run_evaluation_v2_seed42.sh 20
#   ./scripts/run_evaluation_v2_seed42.sh 30
#
# The entry point puts the project's .venv at the front of PATH, so callers need not activate the virtualenv first.
# Before the official run it starts Docker Desktop and Ollama on demand and waits until they are truly usable; the script
# installs no software and does not modify host settings such as Docker Desktop's WSL Integration.
#
# Host-specific paths and run identifiers can be overridden via environment variables:
#   BATCH_ID                batch ID; includes the chosen task count by default
#   RESUME_BATCH            set to 1 to continue an existing BATCH_ID, skipping tasks already fully persisted
#   TASK_SEED               internal task-set seed; 42 by default, and the new 30-task seed 43 is also supported
#   SWEBENCH_ROOT           SWE-bench repo; defaults to $HOME/src/SWE-bench
#   LOCAL_MODEL_BASE_URL    Ollama address; defaults to http://localhost:11434

set -Eeuo pipefail

# Accept only task-set sizes already frozen and verified in the repo, so a typo cannot point to a nonexistent config.
TASK_COUNT="${1:-20}"
case "${TASK_COUNT}" in
    15|20|30) ;;
    *)
        printf 'Usage: %s {15|20|30}\n' "$0" >&2
        exit 2
        ;;
esac

TASK_SEED="${TASK_SEED:-42}"
case "${TASK_SEED}:${TASK_COUNT}" in
    42:15|42:20|42:30|43:30) ;;
    *)
        printf 'Unsupported task-set combination: seed=%s tasks=%s\n' "${TASK_SEED}" "${TASK_COUNT}" >&2
        exit 2
        ;;
esac

# Locate the project from the script's own location so users need not run from the repo root.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
CONFIG_PATH="${PROJECT_ROOT}/configs/evaluation_v2_${TASK_COUNT}_seed${TASK_SEED}.yaml"
TASKS_PATH="${PROJECT_ROOT}/prepared/evaluation_tasks_${TASK_COUNT}_seed${TASK_SEED}.jsonl"
SWEBENCH_ROOT="${SWEBENCH_ROOT:-${HOME}/src/SWE-bench}"
# The new cache/total-budget protocol changes the config fingerprint, so the default batch name must stay isolated from old results;
# when BATCH_ID is set explicitly, the caller's experiment naming is still respected.
BATCH_ID="${BATCH_ID:-evaluation-v2-qwen35-${TASK_COUNT}-seed${TASK_SEED}-budget-cache-v1}"
LOCAL_MODEL_BASE_URL="${LOCAL_MODEL_BASE_URL:-http://localhost:11434}"
RESUME_BATCH="${RESUME_BATCH:-0}"

case "${RESUME_BATCH}" in
    0) RESUME_ARGS=() ;;
    1) RESUME_ARGS=(--resume) ;;
    *)
        printf 'RESUME_BATCH may only be 0 or 1, current value: %s\n' "${RESUME_BATCH}" >&2
        exit 2
        ;;
esac

fail() {
    # All precheck errors share one format and guarantee no experiment directory is created after a failure.
    printf 'Startup check failed: %s\n' "$1" >&2
    exit 1
}

ollama_ready() {
    # Use the project Python to probe the API so the host need not install curl.
    "${PYTHON_BIN}" -c '
import sys
import urllib.request

base_url = sys.argv[1].rstrip("/")
with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
    if response.status != 200:
        raise SystemExit(1)
' "${LOCAL_MODEL_BASE_URL}" >/dev/null 2>&1
}

wait_for_docker() {
    # Docker Desktop usually takes tens of seconds to start; checking both the command and the daemon covers
    # both the state where WSL Integration has not yet mounted the client and the state where the daemon is not ready.
    local attempt
    for ((attempt = 1; attempt <= 45; attempt++)); do
        if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}

wait_for_ollama() {
    local attempt
    for ((attempt = 1; attempt <= 30; attempt++)); do
        if ollama_ready; then
            return 0
        fi
        sleep 2
    done
    return 1
}

start_docker_if_needed() {
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        return
    fi

    printf 'Docker not ready yet, starting Docker Desktop...\n'
    if command -v powershell.exe >/dev/null 2>&1; then
        # WSL cannot manage Windows services directly; launch the installed Desktop via Windows,
        # then keep waiting on docker info rather than assuming the process appearing means it is usable.
        powershell.exe -NoProfile -NonInteractive -Command \
            'Start-Process "$Env:ProgramFiles\Docker\Docker\Docker Desktop.exe"' \
            >/dev/null 2>&1 || true
    fi

    if ! wait_for_docker; then
        cat >&2 <<'EOF'
Startup check failed: Docker daemon is still unreachable after starting Docker Desktop.

Confirm Docker Desktop is installed on Windows and enable the current distro at:
  Settings -> Resources -> WSL Integration
EOF
        exit 1
    fi
    printf 'Docker is ready.\n'
}

start_ollama_if_needed() {
    if ollama_ready; then
        return
    fi

    printf 'Ollama not ready yet, starting the service...\n'
    if command -v ollama >/dev/null 2>&1; then
        # Logs go to /tmp so host service logs are not written into the experiment repo or committed to Git.
        nohup ollama serve >/tmp/local-swe-agent-eval-ollama.log 2>&1 &
    elif command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -NonInteractive -Command \
            'Start-Process -WindowStyle Hidden ollama -ArgumentList "serve"' \
            >/dev/null 2>&1 || true
    else
        fail "Cannot find ollama or powershell.exe, so Ollama cannot be started automatically."
    fi

    if ! wait_for_ollama; then
        fail "Still cannot reach ${LOCAL_MODEL_BASE_URL} after starting Ollama; check the Ollama install and service logs."
    fi
    printf 'Ollama is ready.\n'
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
    fail "Cannot find the project virtualenv ${PYTHON_BIN}; create it and install project dependencies first."
fi

# A Bash script cannot modify the parent shell that invoked it, but putting .venv/bin at the front of the
# current script's PATH is equivalent to activating the env for this evaluation and still works when subprocesses call `python`.
export PATH="${PROJECT_ROOT}/.venv/bin:${PATH}"

if [[ ! -f "${CONFIG_PATH}" || ! -f "${TASKS_PATH}" ]]; then
    fail "The ${TASK_COUNT}-task config or prepared task file does not exist; finish data preparation first."
fi

start_docker_if_needed

if ! command -v claude >/dev/null 2>&1; then
    fail "Cannot find the claude command; install Claude Code and confirm it is on PATH."
fi

if [[ ! -x "${SWEBENCH_ROOT}/.venv/bin/swebench" ]]; then
    fail "Cannot find ${SWEBENCH_ROOT}/.venv/bin/swebench; check SWEBENCH_ROOT or install SWE-bench."
fi

start_ollama_if_needed

printf 'Startup checks passed, starting the %s-task batch: %s\n' "${TASK_COUNT}" "${BATCH_ID}"

# Python's module entry point relies on the repo root being on the import path; switching directory explicitly here ensures
# consistent behavior when users invoke this script from any working directory.
cd -- "${PROJECT_ROOT}"

# exec lets the batch take over the current terminal directly, so Ctrl-C and exit codes pass accurately to the caller.
exec "${PYTHON_BIN}" -m scripts.run_batch \
    --config "${CONFIG_PATH}" \
    --tasks "${TASKS_PATH}" \
    --batch-id "${BATCH_ID}" \
    --swebench-root "${SWEBENCH_ROOT}" \
    --base-url "${LOCAL_MODEL_BASE_URL}" \
    --expected-tasks "${TASK_COUNT}" \
    "${RESUME_ARGS[@]}" \
    --allow-network-preparation
