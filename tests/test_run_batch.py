"""验证批量脚本的固定顺序、可选题数约束和 predictions 合并。"""

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


def test_resume_manifest_accepts_only_complete_contiguous_runs(tmp_path: Path) -> None:
    """断点恢复只能跳过 manifest 已登记且 prediction 完整落盘的连续题目。"""

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
    """配置 fingerprint 变化后不得复用旧批次，以免结果失去可比性。"""

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
    """中断题的旧目录必须保留，并为重跑分配递增的 retry ID。"""

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
    """汇总阶段中断后，同一 harness 已导入的结果应跳过而不是报覆盖错误。"""

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
