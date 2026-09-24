# Local SWE-Agent Evaluation

This repository implements a reproducible, fully local evaluation pipeline for
running an open-weight language model with a coding agent on SWE-bench Verified.

## Phase 1: task loading and repository preparation

The first phase establishes two trusted boundaries:

1. `benchmark.swebench_loader` validates dataset records and exposes only the
   fields an agent is allowed to see.
2. `benchmark.repo_manager` creates a shared repository cache and checks out an
   isolated worktree at the task's exact `base_commit`.

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
heartbeats in an interactive terminal, the live SWE-bench harness stream, and a
final aligned result table.

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

## Phase 5: bounded two-session agent architecture

The completed ten-task evaluation remains the immutable baseline represented by
`configs/evaluation.yaml`. Architecture experiments use `configs/dev_v2.yaml`
and must not replace the baseline result.

Dev v2 gives the same local model a 40-turn hard limit split across two fresh
Claude Code sessions:

1. The implementation session receives 30 turns to reproduce, locate, and leave
   a concrete candidate patch in the worktree.
2. The verification session receives 10 reserved turns, re-reads the full task,
   inspects the existing diff, runs a focused visible test, and repairs the
   first observed failure.

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
