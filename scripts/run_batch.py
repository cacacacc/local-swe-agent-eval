"""Serially solve the frozen tasks in the config, then run batch official evaluation and import results per task."""

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
    """Atomically write batch state, avoiding a half-written JSON if the process fails."""

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
    """Load and strictly validate the checkpoint batch, preventing artifacts from other experiments from mixing into the current run.

    Resume only trusts consecutive task records that have been atomically written to ``batch.json``.
    Even if a task left a partial directory at interruption time, incomplete artifacts are not
    treated as complete; the task is later rerun with a new retry run ID, preserving the
    previous state without overwriting user data.
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

    # Only accept consecutive records starting from task 1, avoiding order misalignment
    # caused by skipped tasks, duplicated tasks, or manually edited manifests; each
    # prediction must also exist before the batch evaluation input can be safely rebuilt.
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
    """Choose a run ID for an incomplete task that does not overwrite the previous state."""

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
    """Select tasks in the order of the frozen ID file and optionally validate the caller-declared task count.

    The frozen ID file itself is the authoritative source for the experiment task set,
    so it is not assumed to be exactly ten tasks by default. ``expected_count`` only
    serves as an extra anti-mistake boundary explicitly requested by the user.
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
    """Merge per-task JSONL and verify each file contains exactly one JSON object."""

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
    """Read the imported per-task result and generate the minimal stable fields needed for the batch summary."""

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
    """Check whether the same harness's official result has been imported, supporting continuation at the summary stage."""

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
    """Run the frozen task-set pipeline and return the batch summary path containing the main metrics."""

    if not _SAFE_BATCH_ID.fullmatch(arguments.batch_id):
        raise AutomatedRunError("--batch-id contains unsafe characters")
    if arguments.expected_tasks is not None and arguments.expected_tasks <= 0:
        raise AutomatedRunError("expected task count must be positive")
    if arguments.evaluation_timeout <= 0:
        raise AutomatedRunError("evaluation timeout must be positive")

    config = ExperimentConfig.load(arguments.config)
    # Formal batch runs represent the primary experiment results and must not be started by mistake while Dev parameters are still mutable.
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
        "Local SWE-bench formal batch experiment",
        f"batch={arguments.batch_id}  tasks={len(tasks)}  model={config.model.name}",
    )
    # Prepare all images once before creating batch artifacts and running the first Agent.
    # A missing image then only fails environment preparation, without leaving a half-run
    # experiment where "the first four tasks finished and the fifth is stuck".
    with reporter.activity("Precheck and download all SWE-bench test images as needed"):
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
        reporter.line(f"  ↻  Resuming from task {len(manifest['runs']) + 1}")
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

    reporter.stage(1, 3, "Run local Agent serially")
    run_paths = [Path(entry["run_path"]) for entry in manifest["runs"]]
    prediction_paths = [run_path / "prediction.jsonl" for run_path in run_paths]
    completed_count = len(run_paths)
    try:
        for index, task in enumerate(tasks, start=1):
            if index <= completed_count:
                reporter.task(index, len(tasks), f"{task.instance_id} (completed, skipped)")
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
            with reporter.activity("Prepare worktree"):
                prepared = manager.prepare(
                    task,
                    allow_network=arguments.allow_network_preparation,
                )
            with reporter.activity("Claude Code solving"):
                run_path = run_claude_task(
                    task,
                    prepared.path,
                    config,
                    config.project_root / config.storage.runs / run_id,
                    base_url=arguments.base_url,
                    workspace_base_commit=prepared.workspace_base_commit,
                )
            # An empty patch must still enter the formal harness so the empty patch rate is counted honestly.
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
        reporter.line(f"\n  ✓ Merged {len(tasks)} predictions: {aggregate_predictions}")

        reporter.stage(2, 3, "Run batch official evaluation with Docker workers")
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
        with reporter.activity(f"Official evaluation ({workers} workers)"):
            run_harness(command, swebench_root, batch_path / "official_evaluation.log")

        report_path = (
            swebench_root / "logs" / "evaluation" / harness_run_id / "results.json"
        )
        if not report_path.is_file():
            raise AutomatedRunError(f"official report is missing: {report_path}")

        reporter.stage(3, 3, "Import and summarize official verdicts per task")
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
        # Ctrl-C is a deliberate user pause, not an experiment failure. The current task
        # counts as complete only after both the prediction and manifest are atomically
        # written to disk; otherwise the next run restarts from that task with a retry ID.
        manifest["status"] = "interrupted"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = "KeyboardInterrupt: interrupted by user"
        _write_json(batch_path / "batch.json", manifest)
        reporter.line(
            f"\n  Checkpoint saved: {len(manifest['runs'])}/{len(tasks)} tasks completed; "
            "continue with the same batch ID and --resume."
        )
        raise
    except Exception as error:
        manifest["status"] = "failed"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_json(batch_path / "batch.json", manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    """Declare formal batch run parameters; the task count defaults to the frozen list and workers to the experiment config."""

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
        help="Optional task-count assertion; when omitted, the frozen task list in the config is authoritative",
    )
    parser.add_argument("--evaluation-timeout", type=int, default=1800)
    parser.add_argument("--evaluation-workers", type=int)
    parser.add_argument("--allow-network-preparation", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the same batch ID, skipping tasks already fully written in batch.json",
    )
    return parser


def main() -> int:
    """Run the formal batch experiment and print the final summary.json path."""

    summary_path = run_batch(build_parser().parse_args())
    print(f"batch_summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
