"""验证分阶段 Agent 的交付物门禁和失败分类。"""

from pathlib import Path

from agent.claude_runner import ClaudeCodeResult
from agent.test_plan import TestPlanRequest as StructuredTestPlanRequest
from agent.test_sandbox import VisibleTestResult
from benchmark.task import SWEbenchTask
from experiment.config import ExperimentConfig
from scripts.run_claude import (
    ScheduledTestEvidence,
    _apply_patch_gate,
    _execute_scheduled_test,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _result(
    exit_code: int = 0,
    test_output: str = "",
    *,
    host_test_calls: int = 0,
    implementation_baseline_test_calls: int = 0,
) -> ClaudeCodeResult:
    """构造不启动真实 Claude Code 的最小结果。"""

    return ClaudeCodeResult(
        exit_code=exit_code,
        agent_log="done\n",
        test_output=test_output,
        events=(),
        timed_out=False,
        metrics={
            "agent_turns": 1,
            "tool_calls": 0,
            "host_test_calls": host_test_calls,
            "implementation_baseline_test_calls": implementation_baseline_test_calls,
            "token_usage": {},
        },
    )


def test_patch_gate_rejects_generic_success_with_empty_diff() -> None:
    """模型正常退出但没有代码修改时，运行状态不得继续伪装成 completed。"""

    validated = _apply_patch_gate(_result(), "\n")

    assert validated.exit_code == 2
    assert validated.metrics["patch_gate"] == {
        "patch_generated": False,
        "existing_source_modified": False,
        "test_attempted": False,
        "visible_test_attempted": False,
        "implementation_baseline_test_attempted": False,
        "host_test_attempted": False,
        "host_test_valid": False,
    }
    assert validated.events[-1]["event_type"] == "patch_validation"


def test_patch_gate_records_test_evidence_without_overriding_cli_failure() -> None:
    """非空补丁和测试证据应被记录，但不能掩盖 Claude CLI 自身失败。"""

    validated = _apply_patch_gate(
        _result(
            exit_code=1,
            test_output="$ pytest\n1 failed\n",
            host_test_calls=1,
        ),
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n",
    )

    assert validated.exit_code == 1
    assert validated.metrics["patch_gate"] == {
        "patch_generated": True,
        "existing_source_modified": True,
        "test_attempted": False,
        "visible_test_attempted": False,
        "implementation_baseline_test_attempted": False,
        "host_test_attempted": True,
        "host_test_valid": False,
    }


def test_patch_gate_counts_authorized_implementation_baseline_test() -> None:
    """Docker helper 应算有效测试尝试，但不得混入宿主测试指标。"""

    validated = _apply_patch_gate(
        _result(implementation_baseline_test_calls=1),
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n",
    )

    assert validated.metrics["patch_gate"]["test_attempted"] is True
    assert (
        validated.metrics["patch_gate"]["implementation_baseline_test_attempted"]
        is True
    )
    assert validated.metrics["patch_gate"]["host_test_attempted"] is False


def test_patch_gate_rejects_only_new_reproduction_files() -> None:
    """只新增复现或临时文件不能冒充对既有产品源码的修复。"""

    patch = (
        "diff --git a/reproduce.py b/reproduce.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n+++ b/reproduce.py\n@@ -0,0 +1 @@\n+print(1)\n"
    )

    validated = _apply_patch_gate(_result(), patch)

    assert validated.exit_code == 2
    assert validated.metrics["patch_gate"]["existing_source_modified"] is False


def test_patch_gate_cannot_be_bypassed_by_modifying_existing_documentation() -> None:
    """修改既有 README 仍不属于产品源码修复，不能绕过补丁门禁。"""

    patch = (
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old\n+new\n"
    )

    validated = _apply_patch_gate(_result(), patch)

    assert validated.exit_code == 2
    assert validated.metrics["patch_gate"]["existing_source_modified"] is False


def test_v2_config_keeps_scheduler_resource_limits() -> None:
    """v2 配置继续固定父进程 Docker 测试的资源边界。"""

    config = ExperimentConfig.load(PROJECT_ROOT / "configs" / "dev_v2.yaml")
    assert config.agent.visible_test_sandbox is True
    assert config.agent.visible_test_timeout_seconds == 900
    assert config.agent.max_tool_output_chars == 12000


def test_scheduler_metrics_require_a_real_sandbox_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """只有沙箱返回结构化结果后，execution 和 passed 指标才能增加。"""

    request = StructuredTestPlanRequest(
        status="accepted",
        target_argv=(
            "python", "-m", "pytest", "tests/test_one.py::test_bug"
        ),
        regression_argv=("python", "-m", "pytest", "tests/test_one.py"),
    )
    monkeypatch.setattr("scripts.run_claude.consume_test_plan", lambda _: request)

    class FakeSandbox:
        """记录 argv 并模拟一次成功的 Docker 测试。"""

        def run(self, repository, *, instance_id, base_commit, command):
            assert repository == tmp_path
            assert instance_id == "owner__repo-7"
            assert base_commit == "a" * 40
            assert command in {request.target_argv, request.regression_argv}
            return VisibleTestResult(
                exit_code=0,
                output="1 passed\n",
                image="swebench/example:latest",
                timed_out=False,
                command_started=True,
            )

    task = SWEbenchTask(
        instance_id="owner__repo-7",
        repo="owner/repo",
        base_commit="b" * 40,
        problem_statement="Fix it.",
    )

    evidence = _execute_scheduled_test(
        tmp_path,
        task=task,
        workspace_base_commit="a" * 40,
        sandbox=FakeSandbox(),
    )

    assert evidence.metrics()["visible_test_requests"] == 1
    assert evidence.metrics()["visible_test_executions"] == 2
    assert evidence.metrics()["visible_test_passed"] == 2
    assert evidence.metrics()["visible_test_rejected"] == 0
    assert evidence.ready_for_verification is True


def test_missing_plan_is_counted_and_blocks_verification() -> None:
    """缺失计划必须形成显式指标，且不能被当作可进入 verification 的证据。"""

    evidence = ScheduledTestEvidence(
        request=StructuredTestPlanRequest(status="missing")
    )

    assert evidence.metrics()["visible_test_missing"] == 1
    assert evidence.metrics()["visible_test_executions"] == 0
    assert evidence.ready_for_verification is False


def test_scheduler_rejects_container_result_before_test_command_started(
    tmp_path: Path,
) -> None:
    """补丁应用等前置步骤失败时，不得冒充真实 Docker 测试或测试超时。"""

    request = StructuredTestPlanRequest(
        status="accepted",
        target_argv=("python", "-m", "pytest", "tests/test_bug.py::test_bug"),
        regression_argv=("python", "-m", "pytest", "tests/test_bug.py"),
    )

    class FakeSandbox:
        """模拟容器已创建但测试命令尚未启动就失败的结果。"""

        def run(self, repository, *, instance_id, base_commit, command):
            return VisibleTestResult(
                exit_code=1,
                output="error: patch failed\n",
                image="swebench/example:latest",
                timed_out=True,
                command_started=False,
            )

    task = SWEbenchTask(
        instance_id="owner__repo-8",
        repo="owner/repo",
        base_commit="b" * 40,
        problem_statement="Fix it.",
    )
    evidence = _execute_scheduled_test(
        tmp_path,
        task=task,
        workspace_base_commit="a" * 40,
        sandbox=FakeSandbox(),
        request=request,
    )

    metrics = evidence.metrics()
    assert metrics["visible_test_executions"] == 0
    assert metrics["visible_test_timed_out"] is False
    assert metrics["visible_test_rejected"] == 1
    assert evidence.ready_for_verification is False
