"""Verify the predictions format, harness command, and empty-patch boundaries of the end-to-end script."""

import json
from pathlib import Path

import pytest

from benchmark.task import SWEbenchTask
from scripts.run_and_evaluate import (
    AutomatedRunError,
    build_harness_command,
    prediction_record,
    write_prediction,
)


def make_task() -> SWEbenchTask:
    """Build a minimal safe task containing no hidden answer fields."""

    return SWEbenchTask(
        instance_id="owner__repo-7",
        repo="owner/repo",
        base_commit="0123456789abcdef0123456789abcdef01234567",
        problem_statement="Fix it.",
    )


def test_write_prediction_uses_saved_patch_and_local_model_label(tmp_path: Path) -> None:
    """The official input must use the saved patch verbatim and convert the local model name to a stable label."""

    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n-old\n+new\n"
    (tmp_path / "patch.diff").write_text(patch, encoding="utf-8")

    prediction_path = write_prediction(tmp_path, make_task(), "qwen3.5:9b")

    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert prediction == {
        "instance_id": "owner__repo-7",
        "model_name_or_path": "local-qwen3.5-9b",
        "model_patch": patch,
    }


def test_write_prediction_rejects_empty_patch(tmp_path: Path) -> None:
    """An empty patch must be rejected before starting an expensive Docker evaluation, while preserving the original run for analysis."""

    (tmp_path / "patch.diff").write_text("\n", encoding="utf-8")

    with pytest.raises(AutomatedRunError, match="generated no patch"):
        write_prediction(tmp_path, make_task(), "qwen3.5:9b")


def test_batch_prediction_can_preserve_empty_patch(tmp_path: Path) -> None:
    """The formal batch evaluation must submit empty patches so the harness can count them in the empty-patch rate."""

    (tmp_path / "patch.diff").write_text("", encoding="utf-8")

    prediction = prediction_record(
        tmp_path,
        make_task(),
        "qwen3.5:9b",
        allow_empty=True,
    )

    assert prediction["model_patch"] == ""


def test_harness_command_fixes_dataset_instance_resources_and_run_id() -> None:
    """The harness command must include the fixed dataset, single-task filter, resource limits, and a unique run ID."""

    command = build_harness_command(
        Path("/opt/swebench"),
        dataset="verified",
        prediction_path=Path("/tmp/prediction.jsonl"),
        instance_ids=("owner__repo-7",),
        workers=2,
        timeout_seconds=1800,
        harness_run_id="evaluation-007",
    )

    assert command == [
        "/opt/swebench",
        "eval",
        "verified",
        "--predictions",
        "/tmp/prediction.jsonl",
        "--workers",
        "2",
        "--timeout",
        "1800",
        "--run-id",
        "evaluation-007",
        "--instance",
        "owner__repo-7",
    ]
