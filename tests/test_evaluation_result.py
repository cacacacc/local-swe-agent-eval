"""验证官方评测结果导入的身份校验、审计字段和防覆盖边界。"""

import hashlib
import json
from pathlib import Path

import pytest

from tracking.evaluation_result import EvaluationImportError, import_official_evaluation


def write_json(path: Path, value: object) -> bytes:
    """用稳定格式写入测试 JSON，并返回用于核对来源哈希的原始字节。"""

    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return raw


def make_run_and_report(tmp_path: Path) -> tuple[Path, Path, bytes]:
    """构造尚未导入官方判定的 run，以及 resolved 的单题 harness 报告。"""

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
    """resolved 判定必须连同 harness 身份和原始报告哈希写入运行产物。"""

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
    """报告未提交当前 instance 时必须失败，避免把别题结果错误关联到该 run。"""

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
    """同一 run 的官方判定一旦落盘就不可被另一次导入静默替换。"""

    run_path, report_path, _ = make_run_and_report(tmp_path)
    arguments = {
        "harness_run_id": "dev-official-1",
        "dataset": "verified",
    }
    import_official_evaluation(run_path, report_path, **arguments)

    with pytest.raises(EvaluationImportError, match="refusing to overwrite"):
        import_official_evaluation(run_path, report_path, **arguments)
