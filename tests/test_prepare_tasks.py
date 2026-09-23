"""验证任务快照只包含 Agent 白名单字段且保持冻结 ID 顺序。"""

import json
from pathlib import Path

import pytest

from benchmark.swebench_loader import SWEbenchLoader
from experiment.config import ExperimentConfig
from scripts.prepare_tasks import _load_instance_ids


def test_load_instance_ids_rejects_duplicates(tmp_path: Path) -> None:
    """重复任务会扭曲样本量，因此必须在下载数据前失败。"""

    path = tmp_path / "ids.json"
    path.write_text('["a", "a"]', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicates"):
        _load_instance_ids(path)


def test_safe_projection_excludes_reference_fields() -> None:
    """即使原始记录含答案字段，序列化载荷也只能保留四个求解字段。"""

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
