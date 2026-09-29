"""Verify that the task snapshot contains only Agent allow-listed fields and keeps the frozen ID order."""

import json
from pathlib import Path

import pytest

from benchmark.swebench_loader import SWEbenchLoader
from experiment.config import ExperimentConfig
from scripts.prepare_tasks import _load_instance_ids


def test_load_instance_ids_rejects_duplicates(tmp_path: Path) -> None:
    """Duplicate tasks distort the sample size, so they must fail before any data is downloaded."""

    path = tmp_path / "ids.json"
    path.write_text('["a", "a"]', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicates"):
        _load_instance_ids(path)


def test_safe_projection_excludes_reference_fields() -> None:
    """Even if the raw record contains answer fields, the serialized payload may only keep the four solving fields."""

    record = {
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "base_commit": "0123456789abcdef0123456789abcdef01234567",
        "problem_statement": "Fix the bug.",
        "patch": "SECRET GOLD PATCH",
        "test_patch": "SECRET HIDDEN TEST",
    }

    task = SWEbenchLoader.from_records([record])[0]
    payload = task.to_agent_payload()

    assert set(payload) == {
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
    }
    assert "SECRET" not in json.dumps(payload)
