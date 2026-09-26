"""验证批量脚本的固定顺序、可选题数约束和 predictions 合并。"""

import json
from pathlib import Path

import pytest

from benchmark.swebench_loader import SWEbenchLoader
from benchmark.task import SWEbenchTask
from scripts.run_and_evaluate import AutomatedRunError
from scripts.run_batch import select_fixed_tasks, write_batch_predictions


def make_snapshot(count: int) -> SWEbenchLoader:
    """构造具有稳定顺序的安全任务快照。"""

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
    """批量次序必须来自冻结 ID 文件，而不是快照文件的偶然行顺序。"""

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
    """未声明题数时应接受任意非空冻结清单，防止批处理被旧十题默认值锁死。"""

    ids_path = tmp_path / "ids.json"
    ids_path.write_text(
        json.dumps([f"owner__repo-{index}" for index in range(15)]),
        encoding="utf-8",
    )

    tasks = select_fixed_tasks(ids_path, make_snapshot(15))

    assert len(tasks) == 15


def test_select_fixed_tasks_rejects_wrong_count(tmp_path: Path) -> None:
    """用户显式声明题数时，缺题或多题必须在启动第一个 Agent 前失败。"""

    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps(["owner__repo-0"]), encoding="utf-8")

    with pytest.raises(AutomatedRunError, match="exactly 10"):
        select_fixed_tasks(ids_path, make_snapshot(1), expected_count=10)


def test_select_fixed_tasks_rejects_empty_list(tmp_path: Path) -> None:
    """空冻结清单必须立即失败，避免最终汇总时发生除零或产生伪批次。"""

    ids_path = tmp_path / "ids.json"
    ids_path.write_text("[]", encoding="utf-8")

    with pytest.raises(AutomatedRunError, match="at least one task"):
        select_fixed_tasks(ids_path, make_snapshot(0))


def test_write_batch_predictions_preserves_one_record_per_task(tmp_path: Path) -> None:
    """合并文件必须保持逐题顺序，并保留代表 Agent 失败的空 patch。"""

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
