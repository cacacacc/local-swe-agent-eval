"""Verify identity validation, audit fields, and anti-overwrite boundaries when importing official evaluation results."""

import hashlib
import json
from pathlib import Path

import pytest

from tracking.evaluation_result import EvaluationImportError, import_official_evaluation


def write_json(path: Path, value: object) -> bytes:
    """Write test JSON in a stable format and return the raw bytes used to verify the source hash."""

    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return raw


def make_run_and_report(tmp_path: Path) -> tuple[Path, Path, bytes]:
    """Build a run that has not yet imported an official verdict, and a single-task harness report marked resolved."""

    run_path = tmp_path / "run" / "owner__repo-1"
    run_path.mkdir(parents=True)
    write_json(run_path / "metadata.json", {"instance_id": "owner__repo-1"})
    write_json(run_path / "result.json", {"official_evaluation": None})
    report_path = tmp_path / "results.json"
    report = {
        "schema_version": 2,
        "submitted_ids": ["owner__repo-1"],
        "resolved_ids": ["owner__repo-1"],
        "unresolved_ids": [],
        "incomplete_ids": [],
        "error_ids": [],
        "infra_failure_ids": [],
        "ambiguous_failure_ids": [],
        "empty_patch_ids": [],
    }
    raw = write_json(report_path, report)
    return run_path, report_path, raw


def test_import_records_resolved_result_and_report_hash(tmp_path: Path) -> None:
    """A resolved verdict must be written to the run artifact together with the harness identity and the raw report hash."""

    run_path, report_path, raw_report = make_run_and_report(tmp_path)

    destination = import_official_evaluation(
        run_path,
        report_path,
        harness_run_id="dev-official-1",
        dataset="verified",
    )

    result = json.loads(destination.read_text(encoding="utf-8"))
    official = result["official_evaluation"]
    assert official["resolved"] is True
    assert official["status"] == "resolved"
    assert official["harness_run_id"] == "dev-official-1"
    assert official["source_report_sha256"] == hashlib.sha256(raw_report).hexdigest()


def test_import_rejects_report_for_another_instance(tmp_path: Path) -> None:
    """A report that does not submit the current instance must fail, to avoid wrongly associating another task's result with this run."""

    run_path, report_path, _ = make_run_and_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["submitted_ids"] = ["other__repo-2"]
    write_json(report_path, report)

    with pytest.raises(EvaluationImportError, match="submitted_ids"):
        import_official_evaluation(
            run_path,
            report_path,
            harness_run_id="wrong-report",
            dataset="verified",
        )


def test_import_refuses_to_overwrite_official_result(tmp_path: Path) -> None:
    """Once an official verdict for a run has been written, another import must not silently replace it."""

    run_path, report_path, _ = make_run_and_report(tmp_path)
    arguments = {
        "harness_run_id": "dev-official-1",
        "dataset": "verified",
    }
    import_official_evaluation(run_path, report_path, **arguments)

    with pytest.raises(EvaluationImportError, match="refusing to overwrite"):
        import_official_evaluation(run_path, report_path, **arguments)


def test_unresolved_test_result_takes_priority_over_ambiguous_diagnostic(
    tmp_path: Path,
) -> None:
    """When a real test failure overlaps with a heuristic diagnostic, it must be classified as unresolved to avoid misreporting an evaluation error."""

    run_path, report_path, _ = make_run_and_report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    instance_id = "owner__repo-1"
    report["resolved_ids"] = []
    report["unresolved_ids"] = [instance_id]
    report["ambiguous_failure_ids"] = [instance_id]
    report["failure_reasons"] = {instance_id: "no_tests_collected"}
    write_json(report_path, report)

    destination = import_official_evaluation(
        run_path,
        report_path,
        harness_run_id="dev-official-overlap",
        dataset="verified",
    )

    official = json.loads(destination.read_text(encoding="utf-8"))[
        "official_evaluation"
    ]
    assert official["status"] == "unresolved"
    assert official["resolved"] is False
