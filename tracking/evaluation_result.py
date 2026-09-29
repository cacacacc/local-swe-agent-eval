"""Validate SWE-bench harness reports and associate official verdicts with existing run artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class EvaluationImportError(RuntimeError):
    """Raised when an official report does not match the run artifacts or would overwrite an existing verdict."""


def _load_mapping(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    """Read a UTF-8 JSON object and keep the raw bytes for provenance-hash auditing."""

    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationImportError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvaluationImportError(f"{label} must contain a JSON object: {path}")
    return value, raw


def _id_set(report: Mapping[str, Any], field: str) -> set[str]:
    """Read the instance ID array from the report, rejecting loose type coercion that would cause mismatches."""

    value = report.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EvaluationImportError(f"official report field {field} must be a string array")
    return set(value)


def import_official_evaluation(
    run_path: Path | str,
    report_path: Path | str,
    *,
    harness_run_id: str,
    dataset: str,
) -> Path:
    """Atomically write a single instance's official verdict into ``result.json``.

    The harness summary report may cover multiple tasks, but the target run's instance must appear in the submitted
    list and in some final classification. The importer does not accept overwrites, so reusing a run ID cannot silently
    tamper with experiment results.
    """

    run_directory = Path(run_path).resolve()
    result_path = run_directory / "result.json"
    metadata_path = run_directory / "metadata.json"
    report_file = Path(report_path).resolve()
    result, _ = _load_mapping(result_path, "run result")
    metadata, _ = _load_mapping(metadata_path, "run metadata")
    report, report_bytes = _load_mapping(report_file, "official report")

    if result.get("official_evaluation") is not None:
        raise EvaluationImportError(
            f"official evaluation already exists; refusing to overwrite: {result_path}"
        )
    instance_id = metadata.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        raise EvaluationImportError("run metadata has no valid instance_id")
    if instance_id not in _id_set(report, "submitted_ids"):
        raise EvaluationImportError(
            f"instance {instance_id} is not present in official report submitted_ids"
        )

    resolved_ids = _id_set(report, "resolved_ids")
    unresolved_ids = _id_set(report, "unresolved_ids")
    incomplete_ids = _id_set(report, "incomplete_ids")
    error_ids = _id_set(report, "error_ids")
    infra_ids = _id_set(report, "infra_failure_ids")
    ambiguous_ids = _id_set(report, "ambiguous_failure_ids")
    empty_patch_ids = _id_set(report, "empty_patch_ids")

    # resolved and unresolved are the official test verdicts, with priority over the harness's extra diagnostic sets.
    # One task may appear in both unresolved_ids and ambiguous_failure_ids (for example when a log heuristic
    # falsely reports no_tests_collected while report.json already records FAIL_TO_PASS); here we must preserve
    # the fact that "tests ran but were not resolved" and must not downgrade it to evaluation_error.
    if instance_id in resolved_ids:
        status = "resolved"
        resolved = True
    elif instance_id in empty_patch_ids:
        status = "invalid_patch"
        resolved = False
    elif instance_id in unresolved_ids:
        status = "unresolved"
        resolved = False
    elif instance_id in incomplete_ids | error_ids | infra_ids | ambiguous_ids:
        status = "evaluation_error"
        resolved = False
    else:
        raise EvaluationImportError(
            f"instance {instance_id} has no final classification in official report"
        )

    result["official_evaluation"] = {
        "schema_version": 1,
        "dataset": dataset,
        "harness_run_id": harness_run_id,
        "instance_id": instance_id,
        "resolved": resolved,
        "status": status,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "source_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "report_schema_version": report.get("schema_version"),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = result_path.with_name(f".{result_path.name}.tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(result_path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise EvaluationImportError(
            f"cannot update run result {result_path}: {error}"
        ) from error
    return result_path
