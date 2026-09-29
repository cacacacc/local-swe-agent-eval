"""Verify the batch script's fixed ordering, optional task-count constraint, and predictions merging."""

import json
from pathlib import Path

import pytest

from benchmark.swebench_loader import SWEbenchLoader
from benchmark.task import SWEbenchTask
from scripts.run_and_evaluate import AutomatedRunError
from scripts.run_batch import (
    _load_resume_manifest,
    _next_attempt_run_id,
    _official_result_is_imported,
    select_fixed_tasks,
    write_batch_predictions,
)


def make_snapshot(count: int) -> SWEbenchLoader:
    """Build a safe task snapshot with stable ordering."""

    return SWEbenchLoader(
        SWEbenchTask(
            instance_id=f"owner__repo-{index}",
            repo="owner/repo",
            base_commit=f"{index:040x}",
            problem_statement=f"Fix issue {index}.",
        )
        for index in range(count)
    )


def test_select_fixed_tasks_uses_frozen_id_order(tmp_path: Path) -> None:
    """The batch order must come from the frozen ID file, not from the incidental line order of the snapshot file."""

    ids_path = tmp_path / "ids.json"
    ids_path.write_text(
        json.dumps(["owner__repo-2", "owner__repo-0", "owner__repo-1"]),
        encoding="utf-8",
    )

    tasks = select_fixed_tasks(ids_path, make_snapshot(3), expected_count=3)

    assert [task.instance_id for task in tasks] == [
        "owner__repo-2",
        "owner__repo-0",
        "owner__repo-1",
    ]


def test_select_fixed_tasks_defaults_to_frozen_list_length(tmp_path: Path) -> None:
    """When no task count is declared, any non-empty frozen list must be accepted, to prevent batching from being locked to the old ten-task default."""

    ids_path = tmp_path / "ids.json"
    ids_path.write_text(
        json.dumps([f"owner__repo-{index}" for index in range(15)]),
        encoding="utf-8",
    )

    tasks = select_fixed_tasks(ids_path, make_snapshot(15))

    assert len(tasks) == 15


def test_select_fixed_tasks_rejects_wrong_count(tmp_path: Path) -> None:
    """When the user declares a task count explicitly, missing or extra tasks must fail before the first Agent starts."""

    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps(["owner__repo-0"]), encoding="utf-8")

    with pytest.raises(AutomatedRunError, match="exactly 10"):
        select_fixed_tasks(ids_path, make_snapshot(1), expected_count=10)


def test_select_fixed_tasks_rejects_empty_list(tmp_path: Path) -> None:
    """An empty frozen list must fail immediately, to avoid division by zero or a spurious batch during final aggregation."""

    ids_path = tmp_path / "ids.json"
    ids_path.write_text("[]", encoding="utf-8")

    with pytest.raises(AutomatedRunError, match="at least one task"):
        select_fixed_tasks(ids_path, make_snapshot(0))


def test_write_batch_predictions_preserves_one_record_per_task(tmp_path: Path) -> None:
    """The merged file must keep per-task order and preserve empty patches that represent Agent failures."""

    paths = []
    for index, patch in enumerate(("diff-one", "")):
        path = tmp_path / f"prediction-{index}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "instance_id": f"owner__repo-{index}",
                    "model_name_or_path": "local-model",
                    "model_patch": patch,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        paths.append(path)

    destination = tmp_path / "predictions.jsonl"
    write_batch_predictions(destination, paths)

    records = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [record["instance_id"] for record in records] == [
        "owner__repo-0",
        "owner__repo-1",
    ]
    assert records[1]["model_patch"] == ""


def test_resume_manifest_accepts_only_complete_contiguous_runs(tmp_path: Path) -> None:
    """Resume may only skip contiguous tasks that are registered in the manifest and whose predictions are fully written."""

    tasks = tuple(make_snapshot(2))
    batch_path = tmp_path / "batch"
    run_path = tmp_path / "runs" / "batch-01" / tasks[0].instance_id
    run_path.mkdir(parents=True)
    (run_path / "result.json").write_text("{}\n", encoding="utf-8")
    (run_path / "prediction.jsonl").write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "batch_id": "batch",
        "harness_run_id": "batch-official",
        "config_fingerprint": "fingerprint",
        "model": "model",
        "instance_ids": [task.instance_id for task in tasks],
        "status": "interrupted",
        "runs": [
            {
                "index": 1,
                "instance_id": tasks[0].instance_id,
                "run_id": "batch-01",
                "run_path": str(run_path),
            }
        ],
    }
    batch_path.mkdir()
    (batch_path / "batch.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    loaded = _load_resume_manifest(
        batch_path,
        batch_id="batch",
        harness_run_id="batch-official",
        config_fingerprint="fingerprint",
        model="model",
        tasks=tasks,
        runs_root=tmp_path / "runs",
    )

    assert len(loaded["runs"]) == 1


def test_resume_manifest_rejects_changed_experiment_identity(tmp_path: Path) -> None:
    """After the config fingerprint changes, an old batch must not be reused, so that results do not lose comparability."""

    tasks = tuple(make_snapshot(1))
    batch_path = tmp_path / "batch"
    batch_path.mkdir()
    (batch_path / "batch.json").write_text(
        json.dumps(
            {
                "batch_id": "batch",
                "harness_run_id": "batch-official",
                "config_fingerprint": "old",
                "model": "model",
                "instance_ids": [tasks[0].instance_id],
                "status": "interrupted",
                "runs": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AutomatedRunError, match="config_fingerprint"):
        _load_resume_manifest(
            batch_path,
            batch_id="batch",
            harness_run_id="batch-official",
            config_fingerprint="new",
            model="model",
            tasks=tasks,
            runs_root=tmp_path / "runs",
        )


def test_retry_run_id_preserves_interrupted_attempt(tmp_path: Path) -> None:
    """The old directory of an interrupted task must be preserved, and a retry run must be assigned an incrementing retry ID."""

    runs_root = tmp_path / "runs"
    workspaces_root = tmp_path / "workspaces"
    (runs_root / "batch-12").mkdir(parents=True)
    (workspaces_root / "batch-12-retry-01").mkdir(parents=True)

    run_id = _next_attempt_run_id(
        "batch-12",
        runs_root=runs_root,
        workspaces_root=workspaces_root,
    )

    assert run_id == "batch-12-retry-02"


def test_resume_skips_official_result_from_same_harness(tmp_path: Path) -> None:
    """After interruption during the aggregation phase, results already imported by the same harness should be skipped rather than raising an overwrite error."""

    run_path = tmp_path / "run"
    run_path.mkdir()
    (run_path / "result.json").write_text(
        json.dumps(
            {
                "official_evaluation": {
                    "harness_run_id": "batch-official",
                    "resolved": True,
                }
            }
        ),
        encoding="utf-8",
    )

    assert _official_result_is_imported(run_path, "batch-official") is True
    with pytest.raises(AutomatedRunError, match="different official"):
        _official_result_is_imported(run_path, "another-official")
