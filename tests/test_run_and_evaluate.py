"""验证端到端脚本的 predictions 格式、harness 命令和空 patch 边界。"""

import json
from pathlib import Path

import pytest

from benchmark.task import SWEbenchTask
from scripts.run_and_evaluate import (
    AutomatedRunError,
    build_harness_command,
    write_prediction,
)


def make_task() -> SWEbenchTask:
    """构造不含任何隐藏答案字段的最小安全任务。"""

    return SWEbenchTask(
        instance_id="owner__repo-7",
        repo="owner/repo",
        base_commit="0123456789abcdef0123456789abcdef01234567",
        problem_statement="Fix it.",
    )


def test_write_prediction_uses_saved_patch_and_local_model_label(tmp_path: Path) -> None:
    """官方输入必须逐字使用已保存 patch，并把本地模型名转换为稳定标签。"""

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
    """空 patch 必须在启动耗时 Docker 评测前被拒绝，并保留原 run 供分析。"""

    (tmp_path / "patch.diff").write_text("\n", encoding="utf-8")

    with pytest.raises(AutomatedRunError, match="generated no patch"):
        write_prediction(tmp_path, make_task(), "qwen3.5:9b")


def test_harness_command_fixes_dataset_instance_resources_and_run_id() -> None:
    """harness 命令必须包含固定数据集、单题过滤、资源限制和唯一 run ID。"""

    command = build_harness_command(
        Path("/opt/swebench"),
        dataset="verified",
        prediction_path=Path("/tmp/prediction.jsonl"),
        instance_id="owner__repo-7",
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
        "--instance",
        "owner__repo-7",
        "--workers",
        "2",
        "--timeout",
        "1800",
        "--run-id",
        "evaluation-007",
    ]
