# Local SWE-Agent Evaluation

This repository implements a reproducible, fully local evaluation pipeline for
running an open-weight language model with a coding agent on SWE-bench Verified.

## Phase 1: task loading and repository preparation

The first phase establishes two trusted boundaries:

1. `benchmark.swebench_loader` validates dataset records and exposes only the
   fields an agent is allowed to see.
2. `benchmark.repo_manager` uses a shared repository cache only as a trusted
   preparation source. It exports the task's exact `base_commit` tree and
   initializes a new one-commit repository with no remote or shared Git object
   database, so the agent cannot inspect post-base fixes through Git history.

The agent-facing task representation intentionally excludes reference patches,
test patches, and other evaluation-only fields.

### Development setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
```

Hugging Face dataset loading is optional:

```bash
python -m pip install -e ".[dev,huggingface]"
```

### Minimal example

```python
from pathlib import Path

from benchmark.repo_manager import RepositoryManager
from benchmark.swebench_loader import SWEbenchLoader

loader = SWEbenchLoader.from_jsonl(Path("tasks.jsonl"))
task = loader.get("django__django-11099")

manager = RepositoryManager(
    cache_root=Path("repo-cache"),
    workspace_root=Path("workspaces"),
)
checkout = manager.prepare(task, allow_network=True)
print(checkout.path)
```

Set `allow_network=False` during a formal offline run. Repository preparation
will then fail rather than silently fetching missing data.

## Phase 2: deterministic mock pipeline

Phase 2 validates orchestration without an LLM. The mock runner creates one
predictable repository change, records only observable events, collects a Git
patch, and writes the complete run artifact set. It deliberately leaves
`official_evaluation` unset because a successful agent process is not evidence
that SWE-bench resolved the issue.

```bash
python -m scripts.run_mock \
  --tasks tasks.jsonl \
  --instance-id django__django-11099 \
  --allow-network
```

Each task receives an immutable directory under `runs/<instance_id>/`. A second
run refuses to overwrite the first so an experimental trajectory cannot be
silently lost.

## Phase 3: prompt and experiment configuration

Experiment behavior is defined by strict YAML files under `configs/`. Unknown
keys, absolute paths, online formal solving, and missing task/prompt files are
rejected before a run starts. Each validated config receives an order-independent
SHA-256 fingerprint that is stored with run metadata.

```bash
python -m scripts.validate_config --config configs/dev.yaml
python -m scripts.validate_config --config configs/evaluation.yaml
```

The Dev configuration intentionally keeps `configuration_frozen: false`. The
Evaluation configuration is frozen only after all three development tasks have
established the final model, prompt, timeout, context, and evaluation policy.

## Phase 4: local Claude Code runner

The real runner invokes Claude Code in non-interactive `stream-json` mode and
routes model requests only to the loopback Ollama endpoint. It fixes the model,
32K context assumption, 8K per-response output limit, 30-turn limit, and wall-clock timeout from the validated
experiment config. User plugins, MCP servers, browser tools, cloud credentials,
and session persistence are disabled for each run. Observable model messages and
tool events are retained, while hidden thinking fields are removed.

The HTTPS proxy environment and disabled web tools provide an auditable layered
egress restriction while preserving HTTP access to Ollama on localhost. They do
not block a program that deliberately opens a direct socket and are not a
kernel-level network namespace. This limitation must be disclosed in the final
report rather than described as absolute network isolation.

```bash
python -m scripts.run_claude \
  --config configs/dev.yaml \
  --tasks prepared/verified_tasks.jsonl \
  --instance-id django__django-11951 \
  --run-id dev-001 \
  --allow-network-preparation
```

The task snapshot must contain only `instance_id`, `repo`, `base_commit`, and
`problem_statement`. Evaluation runs additionally refuse to start until the
evaluation configuration is explicitly frozen.

Create those snapshots during the online preparation phase:

```bash
python -m scripts.prepare_tasks \
  --config configs/dev.yaml \
  --output prepared/dev_tasks.jsonl
python -m scripts.prepare_tasks \
  --config configs/evaluation.yaml \
  --output prepared/evaluation_tasks.jsonl
```

Each real run also records the project commit and dirty state, prompt SHA-256,
Python/Claude/Docker versions, and the exact Ollama model digest. The normalized
values receive a second runtime fingerprint independent of the YAML config
fingerprint.

After the official harness finishes, import its immutable decision into the run
artifact instead of treating the agent's own success message as evidence:

```bash
python -m scripts.import_evaluation \
  --run-path runs/dev-qwen35-003/django__django-11951 \
  --report ~/src/SWE-bench/logs/evaluation/dev-official/results.json \
  --harness-run-id dev-official
```

For subsequent tasks, the complete solve/evaluate/import sequence is available
as one command. Every task still requires a new `run-id`; the script refuses to
overwrite prior workspaces, runs, predictions, or official decisions:

```bash
python -m scripts.run_and_evaluate \
  --config configs/dev.yaml \
  --tasks prepared/dev_tasks.jsonl \
  --instance-id sphinx-doc__sphinx-7440 \
  --run-id dev-qwen35-005-20260924 \
  --swebench-root ~/src/SWE-bench \
  --allow-network-preparation
```

Both automation commands show phase panels, task progress bars, elapsed-time
heartbeats every 30 seconds in an interactive terminal, the live SWE-bench
harness stream, and a final aligned result table.

After Dev is complete and `configs/evaluation.yaml` has been reviewed and frozen,
run all ten fixed evaluation tasks with one command. Agent inference remains
serial to protect GPU memory; the ten predictions are then evaluated together
with the configured two Docker workers:

```bash
python -m scripts.run_batch \
  --config configs/evaluation.yaml \
  --tasks prepared/evaluation_tasks.jsonl \
  --batch-id evaluation-qwen35-20260924 \
  --swebench-root ~/src/SWE-bench \
  --allow-network-preparation
```

The batch stores its frozen task order, per-task run paths, merged predictions,
harness log, and `resolved_count / resolved_rate` under
`runs/batches/<batch-id>/`.

For the fixed 15-task and non-overlapping 20-task seed-42 evaluations, the
repository also provides a local one-command launcher. It selects the project
virtual environment itself and checks Docker, Claude Code, Ollama, and
SWE-bench before creating batch output:

```bash
./scripts/run_evaluation_v2_seed42.sh 15
./scripts/run_evaluation_v2_seed42.sh 20
```

Docker Desktop's WSL integration must still be enabled from Docker Desktop on
Windows; a Linux process inside WSL cannot grant that host-side integration.
The launcher reports the exact setting to change when the Docker CLI is absent.

## Phase 5: bounded three-phase agent architecture

The completed ten-task evaluation remains the immutable baseline represented by
`configs/evaluation.yaml`. Architecture experiments use `configs/dev_v2.yaml`
and must not replace the baseline result.

Dev v2 gives the same local model a 44-turn hard limit split across three fresh
Claude Code sessions:

1. A Bash-free 4-turn planning session uses the issue and unmodified repository
   to select separate target and adjacent-regression argv.
2. The parent runs both commands against the unmodified baseline in cached,
   network-free Docker containers. Only after both really start does the 30-turn
   implementation session receive the issue plus their exit codes and output,
   diagnose the failure, and edit code. The parent reruns the same plan afterward.
3. The 10-turn verification session starts only after both post-implementation
   commands really execute. It receives their bounded evidence, may repair the
   patch, and is followed by one final parent-owned rerun.

Starting a new session prevents implementation history and failed automatic
compaction from consuming the verification context. The v2 prompt also limits
individual source reads to 200 lines and command output to roughly 12,000
characters. Source-read size is currently an auditable prompt constraint; the
visible-test wrapper enforces its output limit in code.

Project tests run through the cached official instance image rather than the
orchestration repository's Python environment. The wrapper applies only the
current Git patch to the image's `/testbed`, starts Docker with `--network none`,
limits CPU, memory, processes, time, and output, and removes the one-shot
container afterward. It never applies the SWE-bench hidden `test_patch` or runs
the official evaluation script during solving. Required images must be pulled
during environment preparation; the wrapper never pulls implicitly.

An empty diff can no longer be reported as a completed agent run. The runner
records a `patch_validation` event, per-phase metrics, test-attempt evidence,
and Claude Code's structured terminal reason.

Validate and run the new Dev architecture with a new run ID:

```bash
python -m scripts.validate_config --config configs/dev_v2.yaml
python -m scripts.run_and_evaluate \
  --config configs/dev_v2.yaml \
  --tasks prepared/dev_tasks.jsonl \
  --instance-id django__django-11951 \
  --run-id dev-v2-django-11951-001 \
  --swebench-root ~/src/SWE-bench
```

After all three Dev v2 tasks have run, `configs/evaluation_v2.yaml` freezes the
same architecture for a ten-task ablation. Its results are reported separately
and never replace the original `configs/evaluation.yaml` baseline. Metrics split
`visible_test_calls` from `host_test_calls`, because prompt compliance alone is
not a reliable sandbox boundary for a small local model.

## Post-r2 hardening for new evaluations

The completed r2 artifacts above remain immutable evidence of the earlier
architecture. New runs use four additional code-enforced boundaries:

1. Agent repositories contain one isolated baseline commit and no remote. The
   upstream commit hash remains in metadata, while patch collection compares
   against the new workspace baseline.
2. A mandatory pre-implementation planner writes a temporary
   `.agent-test-plan.json` containing `target_argv` and `regression_argv`.
   Missing or invalid output stops before implementation instead of silently
   continuing.
3. The parent executes both argv arrays on the unmodified baseline, injects the
   real exit codes and bounded output into implementation, reruns them after the
   patch, and only then opens Bash-disabled verification. The scheduler performs
   one final rerun after any verification repair.
4. Test metrics come from scheduler events rather than command-text matching:
   `visible_test_requests`, `visible_test_missing`, `visible_test_rejected`,
   `visible_test_executions`, `visible_test_passed`, and
   `visible_test_timed_out`. `visible_test_calls` remains a compatibility alias
   for confirmed executions; model-issued test commands are recorded as
   `agent_test_command_calls` and `host_test_calls`.

The pre-implementation gate also enforces test semantics, not only container
startup: the focused target must reproduce the baseline defect with exit code 1,
while the adjacent regression command must pass with exit code 0. The parent
injects a bounded, issue-ranked list of tracked test files because Bash, Glob,
and Grep are unavailable to the planner. After verification, both commands must
exit 0 before the local run can be marked completed.

Patch validation rejects virtual environments, caches, `site-packages`, common
root-level scratch reproductions such as `test_bug.py`, patches larger than 1 MB,
and patches spanning more than 100 files. These delivery gates do not replace the
official harness; they prevent generated artifacts and weak local evidence from
being mislabeled as a clean agent completion.

This hardening is a new experimental protocol. It must not be used to rewrite
the original baseline or r2 results, even when the same task IDs are inspected
for debugging.
