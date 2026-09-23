import json
from pathlib import Path
import subprocess

import pytest

from agent.mock_runner import MockAgentError
from benchmark.task import SWEbenchTask
from scripts.run_mock import run_mock_task
from tracking.run_manager import RunArtifactError


def run_git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def make_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    run_git(repository, "init")
    run_git(repository, "config", "user.name", "Phase Two Test")
    run_git(repository, "config", "user.email", "phase-two@example.invalid")
    (repository / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    run_git(repository, "add", "module.py")
    run_git(repository, "commit", "-m", "base")
    return repository, run_git(repository, "rev-parse", "HEAD")


def make_task(commit: str) -> SWEbenchTask:
    return SWEbenchTask(
        instance_id="example__project-456",
        repo="example/project",
        base_commit=commit,
        problem_statement="Exercise the mock pipeline.",
    )


def test_mock_pipeline_writes_complete_non_evaluation_run(tmp_path) -> None:
    repository, commit = make_repository(tmp_path)
    task = make_task(commit)

    run_path = run_mock_task(task, repository, tmp_path / "runs")

    expected_files = {
        "metadata.json",
        "prompt.txt",
        "agent.log",
        "trajectory.json",
        "patch.diff",
        "test_output.log",
        "result.json",
    }
    assert {path.name for path in run_path.iterdir()} == expected_files

    metadata = json.loads((run_path / "metadata.json").read_text(encoding="utf-8"))
    result = json.loads((run_path / "result.json").read_text(encoding="utf-8"))
    trajectory = json.loads((run_path / "trajectory.json").read_text(encoding="utf-8"))
    patch = (run_path / "patch.diff").read_text(encoding="utf-8")

    assert metadata["status"] == "completed"
    assert metadata["model"] == "mock-no-llm"
    assert metadata["event_count"] == 3
    assert result == {
        "schema_version": 1,
        "run_status": "completed",
        "agent_exit_code": 0,
        "patch_generated": True,
        "official_evaluation": None,
    }
    assert "mock_agent_change.txt" in patch
    assert task.instance_id in patch
    assert [event["event_type"] for event in trajectory["events"]] == [
        "file_write",
        "test_run",
        "agent_exit",
    ]
    assert "reasoning" not in json.dumps(trajectory).lower()
    assert "chain_of_thought" not in json.dumps(trajectory).lower()


def test_mock_pipeline_refuses_to_overwrite_prior_run(tmp_path) -> None:
    repository, commit = make_repository(tmp_path)
    task = make_task(commit)
    runs_root = tmp_path / "runs"
    run_mock_task(task, repository, runs_root)

    with pytest.raises(RunArtifactError, match="refusing to overwrite"):
        run_mock_task(task, repository, runs_root)


def test_mock_pipeline_records_agent_failure(tmp_path) -> None:
    repository, commit = make_repository(tmp_path)
    task = make_task(commit)
    (repository / "mock_agent_change.txt").write_text(
        "pre-existing user content\n",
        encoding="utf-8",
    )
    runs_root = tmp_path / "failed-runs"

    with pytest.raises(MockAgentError, match="refusing to overwrite"):
        run_mock_task(task, repository, runs_root)

    run_path = runs_root / task.instance_id
    metadata = json.loads((run_path / "metadata.json").read_text(encoding="utf-8"))
    result = json.loads((run_path / "result.json").read_text(encoding="utf-8"))
    trajectory = json.loads((run_path / "trajectory.json").read_text(encoding="utf-8"))

    assert metadata["status"] == "failed"
    assert result["agent_exit_code"] == 1
    assert result["official_evaluation"] is None
    assert trajectory["events"][0]["event_type"] == "agent_error"
