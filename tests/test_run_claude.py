"""验证分阶段 Agent 的交付物门禁和失败分类。"""

from pathlib import Path

from agent.claude_runner import ClaudeCodeResult
from benchmark.task import SWEbenchTask
from experiment.config import ExperimentConfig
from scripts.run_claude import _apply_patch_gate, _visible_test_command


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _result(exit_code: int = 0, test_output: str = "") -> ClaudeCodeResult:
    """构造不启动真实 Claude Code 的最小结果。"""

    return ClaudeCodeResult(
        exit_code=exit_code,
        agent_log="done\n",
        test_output=test_output,
        events=(),
        timed_out=False,
        metrics={"agent_turns": 1, "tool_calls": 0, "token_usage": {}},
    )


def test_patch_gate_rejects_generic_success_with_empty_diff() -> None:
    """模型正常退出但没有代码修改时，运行状态不得继续伪装成 completed。"""

    validated = _apply_patch_gate(_result(), "\n")

    assert validated.exit_code == 2
    assert validated.metrics["patch_gate"] == {
        "patch_generated": False,
        "test_attempted": False,
        "visible_test_attempted": False,
        "host_test_attempted": False,
    }
    assert validated.events[-1]["event_type"] == "patch_validation"


def test_patch_gate_records_test_evidence_without_overriding_cli_failure() -> None:
    """非空补丁和测试证据应被记录，但不能掩盖 Claude CLI 自身失败。"""

    validated = _apply_patch_gate(
        _result(exit_code=1, test_output="$ pytest\n1 failed\n"),
        "diff --git a/a.py b/a.py\n",
    )

    assert validated.exit_code == 1
    assert validated.metrics["patch_gate"] == {
        "patch_generated": True,
        "test_attempted": True,
        "visible_test_attempted": False,
        "host_test_attempted": False,
    }


def test_visible_test_command_binds_task_identity_and_resource_limits() -> None:
    """注入模型的沙箱前缀必须固定 instance、base commit 与配置中的资源上限。"""

    task = SWEbenchTask(
        instance_id="owner__repo-7",
        repo="owner/repo",
        base_commit="0123456789abcdef0123456789abcdef01234567",
        problem_statement="Fix it.",
    )
    config = ExperimentConfig.load(PROJECT_ROOT / "configs" / "dev_v2.yaml")

    command = _visible_test_command(task, config)

    assert command is not None
    assert "run_visible_tests.py" in command
    assert "owner__repo-7" in command
    assert task.base_commit in command
    assert "--timeout 900" in command
    assert "--max-output-chars 12000" in command
    assert command.endswith(" --")
