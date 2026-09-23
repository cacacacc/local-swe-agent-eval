"""Run one deterministic mock task through the Phase 2 artifact pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from agent.mock_runner import MockAgentRunner
from benchmark.repo_manager import RepositoryManager
from benchmark.swebench_loader import SWEbenchLoader
from benchmark.task import SWEbenchTask
from tracking.run_manager import RunManager


MOCK_PROMPT = """Phase 2 mock run.
Create one deterministic marker file and emit only observable actions.
Do not access the network.
"""


def run_mock_task(
    task: SWEbenchTask,
    repository: Path | str,
    runs_root: Path | str,
) -> Path:
    """Execute the mock runner and return its completed artifact directory."""

    session = RunManager(runs_root).start(
        task,
        phase="mock",
        agent="MockAgentRunner",
        model="mock-no-llm",
        prompt=MOCK_PROMPT,
    )
    try:
        result = MockAgentRunner().run(task, repository)
        patch = session.collect_patch(repository)
    except Exception as error:
        try:
            partial_patch = session.collect_patch(repository)
        except Exception as patch_error:
            partial_patch = ""
            patch_note = f"Patch collection also failed: {patch_error}\n"
        else:
            patch_note = ""
        session.finalize(
            exit_code=1,
            agent_log=(
                f"Mock agent failed with {type(error).__name__}: {error}\n"
                f"{patch_note}"
            ),
            test_output="",
            events=[
                {
                    "sequence": 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "agent_error",
                    "details": {
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                }
            ],
            patch=partial_patch,
        )
        raise
    session.finalize(
        exit_code=result.exit_code,
        agent_log=result.agent_log,
        test_output=result.test_output,
        events=[event.to_dict() for event in result.events],
        patch=patch,
    )
    return session.path


def _load_tasks(path: Path) -> SWEbenchLoader:
    if path.suffix.lower() == ".jsonl":
        return SWEbenchLoader.from_jsonl(path)
    if path.suffix.lower() == ".json":
        return SWEbenchLoader.from_json(path)
    raise ValueError("--tasks must point to a .json or .jsonl file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("repo-cache"))
    parser.add_argument("--workspace-root", type=Path, default=Path("workspaces"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow repository clone/fetch during environment preparation.",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    task = _load_tasks(arguments.tasks).get(arguments.instance_id)
    prepared = RepositoryManager(
        arguments.cache_root,
        arguments.workspace_root,
    ).prepare(task, allow_network=arguments.allow_network)
    run_path = run_mock_task(task, prepared.path, arguments.runs_root)
    print(run_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
