import json

import pytest

from benchmark.swebench_loader import DatasetFormatError, SWEbenchLoader
from benchmark.task import SWEbenchTask


VALID_RECORD = {
    "instance_id": "example__project-123",
    "repo": "example/project",
    "base_commit": "0123456789abcdef0123456789abcdef01234567",
    "problem_statement": "Fix the edge case.",
    "patch": "SECRET REFERENCE PATCH",
    "test_patch": "SECRET TEST PATCH",
    "FAIL_TO_PASS": '["test_hidden"]',
}


def test_agent_payload_excludes_evaluation_only_fields() -> None:
    task = SWEbenchTask.from_record(VALID_RECORD)

    assert task.to_agent_payload() == {
        "instance_id": VALID_RECORD["instance_id"],
        "repo": VALID_RECORD["repo"],
        "base_commit": VALID_RECORD["base_commit"],
        "problem_statement": VALID_RECORD["problem_statement"],
    }
    assert "SECRET" not in repr(task)


def test_jsonl_loader_preserves_order_and_selects_by_id(tmp_path) -> None:
    second = {
        **VALID_RECORD,
        "instance_id": "example__project-456",
        "base_commit": "abcdef0123456789abcdef0123456789abcdef01",
    }
    source = tmp_path / "tasks.jsonl"
    source.write_text(
        "\n".join(json.dumps(record) for record in (VALID_RECORD, second)),
        encoding="utf-8",
    )

    loader = SWEbenchLoader.from_jsonl(source)
    selected = loader.select([second["instance_id"], VALID_RECORD["instance_id"]])

    assert [task.instance_id for task in loader] == [
        VALID_RECORD["instance_id"],
        second["instance_id"],
    ]
    assert [task.instance_id for task in selected] == [
        second["instance_id"],
        VALID_RECORD["instance_id"],
    ]


def test_loader_rejects_duplicate_instance_ids() -> None:
    with pytest.raises(DatasetFormatError, match="duplicate instance_id"):
        SWEbenchLoader.from_records([VALID_RECORD, VALID_RECORD])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repo", "not-a-repository"),
        ("base_commit", "not-a-commit"),
        ("problem_statement", ""),
    ],
)
def test_loader_rejects_invalid_required_values(field, value) -> None:
    record = {**VALID_RECORD, field: value}
    with pytest.raises(DatasetFormatError):
        SWEbenchLoader.from_records([record])

