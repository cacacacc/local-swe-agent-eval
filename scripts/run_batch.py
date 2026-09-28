"""串行求解配置中冻结的题目，再批量官方评测并逐题导回结果。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Sequence

from benchmark.repo_manager import RepositoryManager
from benchmark.swebench_loader import SWEbenchLoader
from benchmark.task import SWEbenchTask
from experiment.config import ExperimentConfig
from scripts.run_and_evaluate import (
    AutomatedRunError,
    _load_tasks,
    build_harness_command,
    run_harness,
    write_prediction,
)
from scripts.run_claude import prepare_visible_test_image, run_claude_task
from tracking.console import ConsoleReporter
from tracking.evaluation_result import import_official_evaluation


_SAFE_BATCH_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _write_json(path: Path, value: Any) -> None:
    """原子写入批次状态，进程异常时避免留下半截 JSON。"""

    temporary = path.with_name(f".{path.name}.tmp")
    serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def _load_resume_manifest(
    batch_path: Path,
    *,
    batch_id: str,
    harness_run_id: str,
    config_fingerprint: str,
    model: str,
    tasks: Sequence[SWEbenchTask],
    runs_root: Path,
) -> dict[str, Any]:
    """加载并严格校验断点批次，防止把其他实验的产物混入当前运行。

    Resume 只信任已经原子写入 ``batch.json`` 的连续题目记录。即使中断时某道题
    留下了部分目录，也不会把不完整产物视为完成；后续会使用新的 retry run ID
    重跑该题，从而保留现场且不覆盖用户数据。
    """

    manifest_path = batch_path / "batch.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AutomatedRunError(
            f"cannot read resumable batch manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise AutomatedRunError("resumable batch manifest must be a JSON object")

    expected = {
        "batch_id": batch_id,
        "harness_run_id": harness_run_id,
        "config_fingerprint": config_fingerprint,
        "model": model,
        "instance_ids": [task.instance_id for task in tasks],
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    ]
    if mismatches:
        raise AutomatedRunError(
            "resume metadata mismatch for: " + ", ".join(mismatches)
        )
    if value.get("status") == "completed":
        raise AutomatedRunError(f"batch is already completed: {batch_path}")
    runs = value.get("runs")
    if not isinstance(runs, list):
        raise AutomatedRunError("resumable batch manifest has invalid runs list")

    # 只接受从第 1 题开始的连续记录，避免跳题、重复题或手工修改 manifest 后产生
    # 顺序错位；每条 prediction 也必须存在，才能安全重建批量评测输入。
    for expected_index, entry in enumerate(runs, start=1):
        if not isinstance(entry, dict) or expected_index > len(tasks):
            raise AutomatedRunError("resumable batch contains an invalid run entry")
        task = tasks[expected_index - 1]
        if (
            entry.get("index") != expected_index
            or entry.get("instance_id") != task.instance_id
        ):
            raise AutomatedRunError(
                f"resumable batch run order mismatch at index {expected_index}"
            )
        run_path_value = entry.get("run_path")
        run_id = entry.get("run_id")
        if not isinstance(run_path_value, str) or not isinstance(run_id, str):
            raise AutomatedRunError(
                f"resumable batch run path is invalid at index {expected_index}"
            )
        run_path = Path(run_path_value).resolve()
        expected_run_path = (runs_root / run_id / task.instance_id).resolve()
        if run_path != expected_run_path:
            raise AutomatedRunError(
                f"resumable batch run path escapes expected location: {run_path}"
            )
        for required_name in ("result.json", "prediction.jsonl"):
            if not (run_path / required_name).is_file():
                raise AutomatedRunError(
                    f"completed run is missing {required_name}: {run_path}"
                )
    return value


def _next_attempt_run_id(
    base_run_id: str,
    *,
    runs_root: Path,
    workspaces_root: Path,
) -> str:
    """为未完成题选择不覆盖旧现场的 run ID。"""

    for attempt in range(100):
        suffix = "" if attempt == 0 else f"-retry-{attempt:02d}"
        candidate = f"{base_run_id}{suffix}"
        if not (runs_root / candidate).exists() and not (
            workspaces_root / candidate
        ).exists():
            return candidate
    raise AutomatedRunError(f"too many interrupted attempts for {base_run_id}")


def select_fixed_tasks(
    ids_path: Path,
    snapshot: SWEbenchLoader,
    *,
    expected_count: int | None = None,
) -> tuple[SWEbenchTask, ...]:
    """按冻结 ID 文件顺序选择任务，并按需校验调用方声明的题数。

    冻结 ID 文件本身是实验题集的权威来源，因此默认不假设固定为十题。
    ``expected_count`` 只作为用户显式要求的额外防误操作边界。
    """

    try:
        raw_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AutomatedRunError(f"cannot read fixed task IDs {ids_path}: {error}") from error
    if not isinstance(raw_ids, list) or not all(isinstance(item, str) for item in raw_ids):
        raise AutomatedRunError("fixed task ID file must contain a JSON string array")
    if not raw_ids:
        raise AutomatedRunError("fixed task ID file must contain at least one task")
    if expected_count is not None and len(raw_ids) != expected_count:
        raise AutomatedRunError(
            f"batch requires exactly {expected_count} fixed tasks, found {len(raw_ids)}"
        )
    if len(set(raw_ids)) != len(raw_ids):
        raise AutomatedRunError("fixed task ID file contains duplicate instance IDs")
    return tuple(snapshot.get(instance_id) for instance_id in raw_ids)


def write_batch_predictions(
    destination: Path,
    prediction_paths: Sequence[Path],
) -> None:
    """合并逐题 JSONL，并验证每个文件恰好包含一个 JSON object。"""

    records: list[dict[str, Any]] = []
    for prediction_path in prediction_paths:
        try:
            lines = [
                line
                for line in prediction_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(lines) != 1:
                raise ValueError("prediction must contain exactly one non-empty line")
            value = json.loads(lines[0])
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise AutomatedRunError(
                f"cannot merge prediction {prediction_path}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise AutomatedRunError(f"prediction is not a JSON object: {prediction_path}")
        records.append(value)

    temporary = destination.with_name(f".{destination.name}.tmp")
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)


def _result_row(run_path: Path, task: SWEbenchTask) -> dict[str, Any]:
    """读取已导回的单题结果，生成批次汇总所需的最小稳定字段。"""

    result_path = run_path / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    official = result.get("official_evaluation")
    if not isinstance(official, dict):
        raise AutomatedRunError(f"official result was not imported: {result_path}")
    return {
        "instance_id": task.instance_id,
        "run_path": str(run_path),
        "agent_status": result.get("run_status"),
        "patch_generated": result.get("patch_generated"),
        "patch_line_count": result.get("patch_line_count"),
        "official_status": official.get("status"),
        "resolved": official.get("resolved"),
        "metrics": result.get("metrics", {}),
    }


def _official_result_is_imported(run_path: Path, harness_run_id: str) -> bool:
    """判断同一 harness 的官方结果是否已导入，支持在汇总阶段继续执行。"""

    result_path = run_path / "result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AutomatedRunError(f"cannot read run result {result_path}: {error}") from error
    if not isinstance(result, dict):
        raise AutomatedRunError(f"run result must be a JSON object: {result_path}")
    official = result.get("official_evaluation")
    if official is None:
        return False
    if not isinstance(official, dict) or official.get("harness_run_id") != harness_run_id:
        raise AutomatedRunError(
            f"run already contains a different official evaluation: {result_path}"
        )
    return True


def run_batch(arguments: argparse.Namespace) -> Path:
    """完成冻结题集流水线，并返回包含主要指标的批次汇总路径。"""

    if not _SAFE_BATCH_ID.fullmatch(arguments.batch_id):
        raise AutomatedRunError("--batch-id contains unsafe characters")
    if arguments.expected_tasks is not None and arguments.expected_tasks <= 0:
        raise AutomatedRunError("expected task count must be positive")
    if arguments.evaluation_timeout <= 0:
        raise AutomatedRunError("evaluation timeout must be positive")

    config = ExperimentConfig.load(arguments.config)
    # 正式批处理代表主要实验结果，不能在 Dev 参数仍可变化时误启动。
    if config.experiment.phase != "evaluation":
        raise AutomatedRunError("batch script requires an evaluation configuration")
    config.require_frozen()
    snapshot = _load_tasks(arguments.tasks)
    tasks = select_fixed_tasks(
        config.tasks_path,
        snapshot,
        expected_count=arguments.expected_tasks,
    )
    harness_run_id = arguments.harness_run_id or f"{arguments.batch_id}-official"
    if not _SAFE_BATCH_ID.fullmatch(harness_run_id):
        raise AutomatedRunError("--harness-run-id contains unsafe characters")

    batch_path = config.project_root / config.storage.runs / "batches" / arguments.batch_id
    resume = bool(getattr(arguments, "resume", False))
    if batch_path.exists() and not resume:
        raise AutomatedRunError(
            f"batch directory already exists; use --resume to continue it: {batch_path}"
        )
    if not batch_path.exists() and resume:
        raise AutomatedRunError(f"cannot resume missing batch directory: {batch_path}")

    reporter = ConsoleReporter()
    reporter.banner(
        "Local SWE-bench 正式批量实验",
        f"batch={arguments.batch_id}  tasks={len(tasks)}  model={config.model.name}",
    )
    # 在创建 batch 产物和运行第一个 Agent 之前一次性准备全部镜像。这样缺失镜像
    # 只会导致环境准备失败，不会留下“前四题完成、第五码住”的半截实验。
    with reporter.activity("预检并按需下载全部 SWE-bench 测试镜像"):
        for task in tasks:
            prepare_visible_test_image(
                task,
                config,
                allow_network_preparation=arguments.allow_network_preparation,
            )
    if resume:
        manifest = _load_resume_manifest(
            batch_path,
            batch_id=arguments.batch_id,
            harness_run_id=harness_run_id,
            config_fingerprint=config.fingerprint,
            model=config.model.name,
            tasks=tasks,
            runs_root=config.project_root / config.storage.runs,
        )
        resume_history = manifest.setdefault("resumed_at", [])
        if not isinstance(resume_history, list):
            raise AutomatedRunError("resumable batch has invalid resumed_at history")
        resume_history.append(datetime.now(timezone.utc).isoformat())
        manifest["status"] = "running"
        manifest["finished_at"] = None
        manifest.pop("error", None)
        reporter.line(f"  ↻ 从第 {len(manifest['runs']) + 1} 题继续")
    else:
        try:
            batch_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise AutomatedRunError(
                f"batch directory already exists; refusing to overwrite: {batch_path}"
            ) from error
        manifest = {
            "schema_version": 1,
            "batch_id": arguments.batch_id,
            "harness_run_id": harness_run_id,
            "config_fingerprint": config.fingerprint,
            "model": config.model.name,
            "instance_ids": [task.instance_id for task in tasks],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "status": "running",
            "runs": [],
        }
    _write_json(batch_path / "batch.json", manifest)

    reporter.stage(1, 3, "串行运行本地 Agent")
    run_paths = [Path(entry["run_path"]) for entry in manifest["runs"]]
    prediction_paths = [run_path / "prediction.jsonl" for run_path in run_paths]
    completed_count = len(run_paths)
    try:
        for index, task in enumerate(tasks, start=1):
            if index <= completed_count:
                reporter.task(index, len(tasks), f"{task.instance_id}（已完成，跳过）")
                continue
            reporter.task(index, len(tasks), task.instance_id)
            base_run_id = f"{arguments.batch_id}-{index:02d}"
            run_id = _next_attempt_run_id(
                base_run_id,
                runs_root=config.project_root / config.storage.runs,
                workspaces_root=config.project_root / config.storage.workspaces,
            )
            manager = RepositoryManager(
                config.project_root / config.storage.repository_cache,
                config.project_root / config.storage.workspaces / run_id,
            )
            with reporter.activity("准备 worktree"):
                prepared = manager.prepare(
                    task,
                    allow_network=arguments.allow_network_preparation,
                )
            with reporter.activity("Claude Code 求解"):
                run_path = run_claude_task(
                    task,
                    prepared.path,
                    config,
                    config.project_root / config.storage.runs / run_id,
                    base_url=arguments.base_url,
                    workspace_base_commit=prepared.workspace_base_commit,
                )
            # 空 patch 仍必须进入正式 harness，才能诚实计入 empty patch rate。
            prediction_path = write_prediction(
                run_path,
                task,
                config.model.name,
                allow_empty=True,
            )
            run_paths.append(run_path)
            prediction_paths.append(prediction_path)
            manifest["runs"].append(
                {
                    "index": index,
                    "instance_id": task.instance_id,
                    "run_id": run_id,
                    "run_path": str(run_path),
                }
            )
            _write_json(batch_path / "batch.json", manifest)

        aggregate_predictions = batch_path / "predictions.jsonl"
        write_batch_predictions(aggregate_predictions, prediction_paths)
        reporter.line(f"\n  ✓ 已合并 {len(tasks)} 条 prediction：{aggregate_predictions}")

        reporter.stage(2, 3, "使用 Docker workers 批量官方评测")
        swebench_root = arguments.swebench_root.resolve()
        executable = arguments.swebench_executable
        if executable is None:
            executable = swebench_root / ".venv" / "bin" / "swebench"
        executable = executable.resolve()
        if not executable.is_file():
            raise AutomatedRunError(f"SWE-bench executable does not exist: {executable}")
        workers = arguments.evaluation_workers or config.evaluation.max_workers
        command = build_harness_command(
            executable,
            dataset=arguments.swebench_dataset,
            prediction_path=aggregate_predictions,
            instance_ids=tuple(task.instance_id for task in tasks),
            workers=workers,
            timeout_seconds=arguments.evaluation_timeout,
            harness_run_id=harness_run_id,
        )
        with reporter.activity(f"官方评测（{workers} workers）"):
            run_harness(command, swebench_root, batch_path / "official_evaluation.log")

        report_path = (
            swebench_root / "logs" / "evaluation" / harness_run_id / "results.json"
        )
        if not report_path.is_file():
            raise AutomatedRunError(f"official report is missing: {report_path}")

        reporter.stage(3, 3, "逐题导回并汇总官方判定")
        rows: list[dict[str, Any]] = []
        for task, run_path in zip(tasks, run_paths):
            if not _official_result_is_imported(run_path, harness_run_id):
                import_official_evaluation(
                    run_path,
                    report_path,
                    harness_run_id=harness_run_id,
                    dataset=config.dataset.name,
                )
            rows.append(_result_row(run_path, task))

        resolved_count = sum(row["resolved"] is True for row in rows)
        summary = {
            "schema_version": 1,
            "batch_id": arguments.batch_id,
            "harness_run_id": harness_run_id,
            "task_count": len(rows),
            "resolved_count": resolved_count,
            "resolved_rate": resolved_count / len(rows),
            "tasks": rows,
        }
        summary_path = batch_path / "summary.json"
        _write_json(summary_path, summary)
        manifest["status"] = "completed"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["summary"] = str(summary_path)
        _write_json(batch_path / "batch.json", manifest)

        reporter.table(
            ("#", "instance", "patch", "official", "resolved"),
            tuple(
                (
                    index,
                    row["instance_id"],
                    row["patch_line_count"],
                    row["official_status"],
                    row["resolved"],
                )
                for index, row in enumerate(rows, start=1)
            ),
        )
        reporter.line(
            f"\nResolved Rate: {resolved_count}/{len(rows)} = "
            f"{summary['resolved_rate']:.1%}"
        )
        reporter.line(f"summary={summary_path}")
        return summary_path
    except KeyboardInterrupt:
        # Ctrl-C 属于用户主动暂停而不是实验失败。当前题只有在 prediction 与 manifest
        # 都已原子落盘后才算完成；否则下次会用 retry ID 从该题重新开始。
        manifest["status"] = "interrupted"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = "KeyboardInterrupt: interrupted by user"
        _write_json(batch_path / "batch.json", manifest)
        reporter.line(
            f"\n  已保存断点：完成 {len(manifest['runs'])}/{len(tasks)} 题；"
            "使用同一 batch ID 和 --resume 可继续。"
        )
        raise
    except Exception as error:
        manifest["status"] = "failed"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_json(batch_path / "batch.json", manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    """声明正式批量运行参数；题数默认取冻结清单，workers 取实验配置。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--swebench-root", type=Path, required=True)
    parser.add_argument("--harness-run-id")
    parser.add_argument("--swebench-dataset", default="verified")
    parser.add_argument("--swebench-executable", type=Path)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument(
        "--expected-tasks",
        type=int,
        help="可选的题数断言；省略时以配置中的冻结任务清单为准",
    )
    parser.add_argument("--evaluation-timeout", type=int, default=1800)
    parser.add_argument("--evaluation-workers", type=int)
    parser.add_argument("--allow-network-preparation", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="继续同一 batch ID，跳过 batch.json 中已经完整落盘的题目",
    )
    return parser


def main() -> int:
    """执行正式批量实验，并打印最终 summary.json 路径。"""

    summary_path = run_batch(build_parser().parse_args())
    print(f"batch_summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
