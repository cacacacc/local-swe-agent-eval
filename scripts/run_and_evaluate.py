"""串行完成本地 Agent 求解、SWE-bench 官方评测和结果导回。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Sequence

from benchmark.repo_manager import RepositoryManager
from benchmark.swebench_loader import SWEbenchLoader
from benchmark.task import SWEbenchTask
from experiment.config import ExperimentConfig
from scripts.run_claude import prepare_visible_test_image, run_claude_task
from tracking.console import ConsoleReporter
from tracking.evaluation_result import import_official_evaluation


class AutomatedRunError(RuntimeError):
    """当自动化流水线无法安全进入下一阶段时抛出。"""


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _load_tasks(path: Path) -> SWEbenchLoader:
    """按照安全快照的扩展名加载任务，不接受原始数据集的额外答案字段。"""

    if path.suffix.lower() == ".jsonl":
        return SWEbenchLoader.from_jsonl(path)
    if path.suffix.lower() == ".json":
        return SWEbenchLoader.from_json(path)
    raise AutomatedRunError("--tasks must point to a .json or .jsonl file")


def prediction_record(
    run_path: Path,
    task: SWEbenchTask,
    model_name: str,
    *,
    allow_empty: bool = False,
) -> dict[str, str]:
    """读取 run patch 并构造 prediction；批量正式评测可显式保留空 patch。"""

    patch_path = run_path / "patch.diff"
    try:
        patch = patch_path.read_text(encoding="utf-8")
    except OSError as error:
        raise AutomatedRunError(f"cannot read generated patch {patch_path}: {error}") from error
    if not patch.strip() and not allow_empty:
        # 空 patch 不能送入 harness 冒充有效候选；Agent 失败现场仍留在 run_path 中。
        raise AutomatedRunError(
            f"agent generated no patch; official evaluation was not started: {run_path}"
        )

    return {
        "instance_id": task.instance_id,
        "model_name_or_path": f"local-{model_name.replace(':', '-')}",
        "model_patch": patch,
    }


def write_prediction(
    run_path: Path,
    task: SWEbenchTask,
    model_name: str,
    *,
    allow_empty: bool = False,
) -> Path:
    """从不可变 run patch 创建官方 harness 接受的单题 JSONL。"""

    prediction = prediction_record(
        run_path,
        task,
        model_name,
        allow_empty=allow_empty,
    )
    destination = run_path / "prediction.jsonl"
    temporary = run_path / ".prediction.jsonl.tmp"
    serialized = json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n"
    try:
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(destination)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise AutomatedRunError(f"cannot write prediction {destination}: {error}") from error
    return destination


def build_harness_command(
    executable: Path,
    *,
    dataset: str,
    prediction_path: Path,
    instance_ids: Sequence[str],
    workers: int,
    timeout_seconds: int,
    harness_run_id: str,
) -> list[str]:
    """构造无 shell 插值的官方评测命令，确保 run ID 与单题过滤器显式固定。"""

    command = [
        str(executable),
        "eval",
        dataset,
        "--predictions",
        str(prediction_path),
        "--workers",
        str(workers),
        "--timeout",
        str(timeout_seconds),
        "--run-id",
        harness_run_id,
    ]
    # 重复 --instance 是 SWE-bench CLI 的官方多题过滤方式；显式列出能够防止
    # predictions 文件意外混入其他题目后扩大评测范围。
    for instance_id in instance_ids:
        command.extend(("--instance", instance_id))
    return command


def run_harness(command: list[str], swebench_root: Path, log_path: Path) -> None:
    """运行官方 harness，同时把输出显示给操作者并完整保存到 run artifact。"""

    lines: list[str] = []
    try:
        process = subprocess.Popen(
            command,
            cwd=swebench_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        raise AutomatedRunError(f"cannot start SWE-bench harness: {error}") from error

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return_code = process.wait()
    log_path.write_text("".join(lines), encoding="utf-8")
    if return_code != 0:
        raise AutomatedRunError(
            f"SWE-bench harness exited with {return_code}; see {log_path}"
        )


def run_pipeline(arguments: argparse.Namespace) -> Path:
    """执行单题完整流水线，并返回已写入官方判定的 result 路径。"""

    if not _SAFE_RUN_ID.fullmatch(arguments.run_id):
        raise AutomatedRunError("--run-id contains unsafe characters")
    harness_run_id = arguments.harness_run_id or f"{arguments.run_id}-eval"
    if not _SAFE_RUN_ID.fullmatch(harness_run_id):
        raise AutomatedRunError("--harness-run-id contains unsafe characters")
    if arguments.evaluation_timeout <= 0:
        raise AutomatedRunError("--evaluation-timeout must be positive")

    reporter = ConsoleReporter()
    config = ExperimentConfig.load(arguments.config)
    if config.experiment.phase == "evaluation":
        config.require_frozen()
    task = _load_tasks(arguments.tasks).get(arguments.instance_id)
    workers = arguments.evaluation_workers or config.evaluation.max_workers
    if workers <= 0:
        raise AutomatedRunError("evaluation workers must be positive")

    reporter.banner(
        "Local SWE-bench 单题流水线",
        f"run={arguments.run_id}  instance={task.instance_id}  model={config.model.name}",
    )
    reporter.stage(1, 4, "准备仓库并运行本地 Agent")
    with reporter.activity("检查并按需下载 SWE-bench 测试镜像"):
        prepare_visible_test_image(
            task,
            config,
            allow_network_preparation=arguments.allow_network_preparation,
        )
    manager = RepositoryManager(
        config.project_root / config.storage.repository_cache,
        config.project_root / config.storage.workspaces / arguments.run_id,
    )
    with reporter.activity("创建干净 worktree"):
        prepared = manager.prepare(
            task,
            allow_network=arguments.allow_network_preparation,
        )
    with reporter.activity("Claude Code 求解"):
        run_path = run_claude_task(
            task,
            prepared.path,
            config,
            config.project_root / config.storage.runs / arguments.run_id,
            base_url=arguments.base_url,
            workspace_base_commit=prepared.workspace_base_commit,
        )

    reporter.stage(2, 4, "生成官方 prediction")
    prediction_path = write_prediction(run_path, task, config.model.name)
    reporter.line(f"  ✓ prediction={prediction_path}")

    swebench_root = arguments.swebench_root.resolve()
    executable = arguments.swebench_executable
    if executable is None:
        executable = swebench_root / ".venv" / "bin" / "swebench"
    executable = executable.resolve()
    if not executable.is_file():
        raise AutomatedRunError(f"SWE-bench executable does not exist: {executable}")

    reporter.stage(3, 4, "运行 SWE-bench Docker harness")
    command = build_harness_command(
        executable,
        dataset=arguments.swebench_dataset,
        prediction_path=prediction_path,
        instance_ids=(task.instance_id,),
        workers=workers,
        timeout_seconds=arguments.evaluation_timeout,
        harness_run_id=harness_run_id,
    )
    with reporter.activity("官方 Docker 评测"):
        run_harness(command, swebench_root, run_path / "official_evaluation.log")
    report_path = swebench_root / "logs" / "evaluation" / harness_run_id / "results.json"
    if not report_path.is_file():
        raise AutomatedRunError(
            f"harness completed but official report is missing: {report_path}"
        )
    reporter.stage(4, 4, "校验并导回官方结果")
    result_path = import_official_evaluation(
        run_path,
        report_path,
        harness_run_id=harness_run_id,
        dataset=config.dataset.name,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    official = result["official_evaluation"]
    reporter.table(
        ("instance", "patch lines", "status", "resolved"),
        (
            (
                task.instance_id,
                result["patch_line_count"],
                official["status"],
                official["resolved"],
            ),
        ),
    )
    return result_path


def build_parser() -> argparse.ArgumentParser:
    """声明求解与官方评测所需参数；所有身份字段必须由操作者显式给出。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--swebench-root", type=Path, required=True)
    parser.add_argument("--harness-run-id")
    parser.add_argument("--swebench-dataset", default="verified")
    parser.add_argument("--swebench-executable", type=Path)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--evaluation-timeout", type=int, default=1800)
    parser.add_argument("--evaluation-workers", type=int)
    parser.add_argument("--allow-network-preparation", action="store_true")
    return parser


def main() -> int:
    """运行自动化流水线并打印最终 result.json 路径。"""

    result_path = run_pipeline(build_parser().parse_args())
    print(f"official_result={result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
