"""Serially complete local Agent solving, official SWE-bench evaluation, and result import."""

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
    """Raised when the automated pipeline cannot safely proceed to the next stage."""


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _load_tasks(path: Path) -> SWEbenchLoader:
    """Load tasks according to the safe snapshot extension; rejects the raw dataset's extra answer fields."""

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
    """Read the run patch and build a prediction; formal batch evaluation may explicitly retain an empty patch."""

    patch_path = run_path / "patch.diff"
    try:
        patch = patch_path.read_text(encoding="utf-8")
    except OSError as error:
        raise AutomatedRunError(f"cannot read generated patch {patch_path}: {error}") from error
    if not patch.strip() and not allow_empty:
        # An empty patch cannot be fed into the harness as a valid candidate; the Agent failure site remains in run_path.
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
    """Create the single-task JSONL accepted by the official harness from the immutable run patch."""

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
    """Build the official evaluation command without shell interpolation, ensuring the run ID and single-task filter are explicitly pinned."""

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
    # Repeating --instance is SWE-bench CLI's official multi-task filter; listing them
    # explicitly prevents the predictions file from accidentally mixing in other tasks and widening evaluation scope.
    for instance_id in instance_ids:
        command.extend(("--instance", instance_id))
    return command


def run_harness(command: list[str], swebench_root: Path, log_path: Path) -> None:
    """Run the official harness while showing output to the operator and saving it fully to the run artifact."""

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
    """Run the complete single-task pipeline and return the result path with the official verdict written in."""

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
        "Local SWE-bench single-task pipeline",
        f"run={arguments.run_id}  instance={task.instance_id}  model={config.model.name}",
    )
    reporter.stage(1, 4, "Prepare repository and run local Agent")
    with reporter.activity("Check and download SWE-bench test images as needed"):
        prepare_visible_test_image(
            task,
            config,
            allow_network_preparation=arguments.allow_network_preparation,
        )
    manager = RepositoryManager(
        config.project_root / config.storage.repository_cache,
        config.project_root / config.storage.workspaces / arguments.run_id,
    )
    with reporter.activity("Create clean worktree"):
        prepared = manager.prepare(
            task,
            allow_network=arguments.allow_network_preparation,
        )
    with reporter.activity("Claude Code solving"):
        run_path = run_claude_task(
            task,
            prepared.path,
            config,
            config.project_root / config.storage.runs / arguments.run_id,
            base_url=arguments.base_url,
            workspace_base_commit=prepared.workspace_base_commit,
        )

    reporter.stage(2, 4, "Generate official prediction")
    prediction_path = write_prediction(run_path, task, config.model.name)
    reporter.line(f"  ✓ prediction={prediction_path}")

    swebench_root = arguments.swebench_root.resolve()
    executable = arguments.swebench_executable
    if executable is None:
        executable = swebench_root / ".venv" / "bin" / "swebench"
    executable = executable.resolve()
    if not executable.is_file():
        raise AutomatedRunError(f"SWE-bench executable does not exist: {executable}")

    reporter.stage(3, 4, "Run SWE-bench Docker harness")
    command = build_harness_command(
        executable,
        dataset=arguments.swebench_dataset,
        prediction_path=prediction_path,
        instance_ids=(task.instance_id,),
        workers=workers,
        timeout_seconds=arguments.evaluation_timeout,
        harness_run_id=harness_run_id,
    )
    with reporter.activity("Official Docker evaluation"):
        run_harness(command, swebench_root, run_path / "official_evaluation.log")
    report_path = swebench_root / "logs" / "evaluation" / harness_run_id / "results.json"
    if not report_path.is_file():
        raise AutomatedRunError(
            f"harness completed but official report is missing: {report_path}"
        )
    reporter.stage(4, 4, "Validate and import official results")
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
    """Declare the parameters required for solving and official evaluation; all identity fields must be provided explicitly by the operator."""

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
    """Run the automated pipeline and print the final result.json path."""

    result_path = run_pipeline(build_parser().parse_args())
    print(f"official_result={result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
