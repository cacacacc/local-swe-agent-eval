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

