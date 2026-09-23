"""验证通用运行产物对超时和补丁规模的结构化分类。"""

import json
from pathlib import Path

from benchmark.task import SWEbenchTask
from tracking.run_manager import RunManager


def test_finalize_classifies_timeout_and_counts_changed_lines(tmp_path: Path) -> None:
    """退出码 124 必须归类为 timeout，且补丁规模不能统计 diff header。"""

    task = SWEbenchTask(
        instance_id="owner__repo-1",
        repo="owner/repo",
        base_commit="0123456789abcdef0123456789abcdef01234567",
        problem_statement="Fix it.",
    )
    session = RunManager(tmp_path / "runs").start(
        task,
        phase="dev",
        agent="claude-code",
        model="qwen2.5-coder:7b",
        prompt="prompt\n",
    )
    session.finalize(
        exit_code=124,
        agent_log="",
        test_output="",
        events=[],
        patch="--- a/file.py\n+++ b/file.py\n-old\n+new\n",
        metrics={"timed_out": True},
    )

    result = json.loads((session.path / "result.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (session.path / "metadata.json").read_text(encoding="utf-8")
    )

    assert result["run_status"] == "timeout"
    assert result["patch_line_count"] == 2
    assert metadata["status"] == "timeout"
    assert metadata["metrics"]["timed_out"] is True
