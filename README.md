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

Both checked-in configurations intentionally have `configuration_frozen: false`.
The evaluation config must be frozen only after development tasks establish the
final model, prompt, timeout, and evaluation policy.

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
