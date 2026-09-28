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

For the fixed, mutually non-overlapping 15-, 20-, and 30-task seed-42 evaluations, the
repository also provides a local one-command launcher. It selects the project
virtual environment itself, starts Docker Desktop and Ollama when necessary,
and checks Claude Code and SWE-bench before creating batch output:

```bash
./scripts/run_evaluation_v2_seed42.sh 15
./scripts/run_evaluation_v2_seed42.sh 20
./scripts/run_evaluation_v2_seed42.sh 30
```

A second non-overlapping 30-task sample is frozen under seed 43. Its configuration,
ID list, and safe agent snapshot are respectively:
`configs/evaluation_v2_30_seed43.yaml`,
`experiments/evaluation_tasks_30_seed43.json`, and
`prepared/evaluation_tasks_30_seed43.jsonl`. It excludes all 78 instances used by
the existing Dev and evaluation sets before sampling 30 of the remaining 422.
Run it with the dedicated short launcher:

```bash
./scripts/run_evaluation_v2_seed43.sh
```

The batch can be paused with `Ctrl-C`. To continue from the first task that was
not fully recorded, reuse the exact same batch ID and set `RESUME_BATCH=1`:

```bash
BATCH_ID=evaluation-v2-qwen35-30-recovery-handoff-v1 \
RESUME_BATCH=1 \
./scripts/run_evaluation_v2_seed42.sh 30
```

Resume validates the config fingerprint, model, frozen task order, and completed
artifacts before skipping work. An interrupted in-progress task is rerun under a
new retry run ID, preserving its partial directory for diagnosis.

Docker Desktop's WSL integration must still be enabled from Docker Desktop on
Windows; a Linux process inside WSL cannot grant that host-side integration.
The launcher can start the installed Windows applications, but it reports the
exact setting to change if WSL still cannot reach the Docker daemon.

## Phase 5: bounded two-session agent architecture

The completed ten-task evaluation remains the immutable baseline represented by
`configs/evaluation.yaml`. Architecture experiments use `configs/dev_v2.yaml`
and must not replace the baseline result.

Dev v2 gives the same local model a 40-turn hard limit split across two fresh
Claude Code sessions, with deterministic parent-owned test planning between them:

1. The implementation session receives 30 turns to investigate the issue and
   produce a candidate patch.
2. The parent maps modified source paths to existing repository tests, selecting
   a target module plus a different adjacent regression module (or a Django test
   app) without asking another model session.
3. The parent runs every command once on the unmodified instance image and once
   with the candidate patch in cached, network-free Docker containers. Only a
   baseline-pass to patched-fail transition is classified as `new_regression`;
   unchanged baseline failures remain diagnostic context.
   Every non-empty patch enters the 10-turn verification session; missing plans,
   runner errors, and failing tests are recorded and injected as evidence instead
   of acting as hard gates. When a `new_regression` exists, Verification receives
   only the first proven regression and the candidate patch, and Claude Code's
   tool whitelist is reduced to `Read,Edit`.
4. If implementation produces no usable patch, the otherwise-unused 10
   verification turns become Recovery. Its first two turns are a mandatory
   Read→Edit gate where `Read,Edit` are the only available tools: the first turn
   reads one handoff-selected source file to satisfy Claude Code's edit prerequisite,
   and the second immediately edits that file. If no existing-source patch appears,
   the remaining eight turns run a no-Bash fallback with targeted read tools. This
   keeps the 40-turn budget fixed while preventing renewed broad exploration.

Starting a new session prevents implementation history and failed automatic
compaction from consuming the verification context. The v2 prompt also limits
individual source reads to 200 lines and command output to roughly 12,000
characters. Source-read size is currently an auditable prompt constraint; the
visible-test wrapper enforces its output limit in code.

Project tests run through the cached official instance image rather than the
orchestration repository's Python environment. Parent-scheduled calls apply the
current Git patch to the image's `/testbed`. The wrapper starts Docker with
`--network none`, limits CPU, memory,
processes, time, and output, and removes the one-shot container afterward. It
never applies the SWE-bench hidden `test_patch` or runs the official evaluation
script during solving. Required images must be pulled during environment
preparation; the wrapper never pulls implicitly.

New v2 runs also impose a 30-minute wall-clock budget across all model and Docker
phases. Target tests retain the 900-second command timeout, while the broader
regression command is limited to 300 seconds. Within one task, unchanged baseline
results are always reused between Initial and Final. A candidate result is reused
only when its patch hash is unchanged and the previous execution genuinely passed;
failed or timed-out candidates are rerun so transient Docker failures are not
silently frozen. Budget exhaustion preserves the patch produced so far for the
official harness instead of discarding the task.

An empty diff can no longer be reported as a completed agent run. Generated
virtual environments and caches are excluded from both visible-test and final
patches without deleting the worktree evidence. The runner records a
`patch_validation` event, per-phase metrics, test-attempt evidence, and Claude
Code's structured terminal reason.

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
2. The parent derives a target command and a genuinely different adjacent
   regression command from the repository and changed source paths. A stale
   model-created control file is consumed only for audit and never controls
   execution.
3. The parent runs both argv arrays against the unchanged baseline and candidate
   patch in the cached, network-free SWE-bench image. Every non-empty patch reaches
   Verification, which receives the comparison classifications and bounded output
   with Bash disabled. A proven `new_regression` selects the separate
   `verification_regression_repair` phase, passes only its first failure, and
   exposes only `Read,Edit`. Missing or failed tests remain metrics rather than
   protocol failures, and the scheduler regenerates commands from the final patch
   before rerunning them.
4. Parent-owned tests use scheduler events, including
   `visible_test_parent_generated`:
   `visible_test_requests`, `visible_test_missing`, `visible_test_rejected`,
   `visible_test_executions`, `visible_test_passed`, and
   `visible_test_timed_out`. `visible_test_calls` remains a compatibility alias
   for confirmed executions; model-issued test commands are recorded as
   `agent_test_command_calls` and `host_test_calls`.
5. Test events additionally record `duration_seconds`, `cache_hit`,
   `baseline_timed_out`, `candidate_timed_out`, and `task_budget_exhausted`.
   Cached executions contribute zero duration in the later phase, while the full
   single-task wall clock remains available as `metadata.runtime_seconds`.

This hardening is a new experimental protocol. It must not be used to rewrite
the original baseline or r2 results, even when the same task IDs are inspected
for debugging.
