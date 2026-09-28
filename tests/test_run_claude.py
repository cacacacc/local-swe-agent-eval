"""验证分阶段 Agent 的交付物门禁和失败分类。"""

import json
from pathlib import Path
import subprocess

from agent.claude_runner import ClaudeCodeResult
from agent.test_plan import TestPlanRequest as StructuredTestPlanRequest
from agent.test_sandbox import VisibleTestResult
from benchmark.task import SWEbenchTask
from experiment.config import ExperimentConfig
from scripts.run_claude import (
    ScheduledTestEvidence,
    _TaskBudget,
    _apply_patch_gate,
    _build_recovery_handoff,
    _build_recovery_source_context,
    _execute_scheduled_test,
    _scheduled_test_result,
    _should_run_verification,
    _verification_policy,
    run_claude_task,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _result(
    exit_code: int = 0,
    test_output: str = "",
    *,
    host_test_calls: int = 0,
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
        "host_test_attempted": False,
        "host_test_valid": False,
    }
    assert validated.events[-1]["event_type"] == "patch_validation"


def test_recovery_handoff_keeps_visible_findings_without_tool_results() -> None:
    """Recovery 只应接收公开结论和工具参数，不能携带 thinking 或工具输出。"""

    implementation = ClaudeCodeResult(
        exit_code=1,
        agent_log="raw log must not be copied",
        test_output="tool output must not be copied",
        events=(
            {
                "event_type": "assistant",
                "details": {
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": "private chain"},
                            {
                                "type": "text",
                                "text": "The likely defect is in parser.py.",
                            },
                            {
                                "type": "tool_use",
                                "name": "Read",
                                "input": {
                                    "file_path": "src/parser.py",
                                    "offset": 10,
                                    "limit": 80,
                                },
                            },
                        ]
                    }
                },
            },
            {
                "event_type": "user",
                "details": {
                    "message": {
                        "content": [
                            {"type": "tool_result", "content": "secret result"}
                        ]
                    }
                },
            },
        ),
        timed_out=False,
        metrics={
            "terminal_reason": "blocking_limit",
            "result_subtype": "error_during_execution",
        },
    )

    handoff = _build_recovery_handoff(implementation, maximum_chars=2000)

    assert "reason=blocking_limit" in handoff
    assert "The likely defect is in parser.py." in handoff
    assert "Read: file_path=src/parser.py, offset=10, limit=80" in handoff
    assert "private chain" not in handoff
    assert "secret result" not in handoff
    assert "raw log" not in handoff
    assert "tool output" not in handoff


def test_recovery_source_context_reads_only_recent_repository_source(
    tmp_path: Path,
) -> None:
    """强制 Edit 上下文只能包含仓库内最近读取的受信任源码片段。"""

    source = tmp_path / "src" / "parser.py"
    source.parent.mkdir()
    source.write_text("first = 1\nsecond = 2\nthird = 3\n", encoding="utf-8")
    outside = tmp_path.parent / "outside.py"
    implementation = ClaudeCodeResult(
        exit_code=1,
        agent_log="",
        test_output="",
        events=(
            {
                "event_type": "assistant",
                "details": {
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Read",
                                "input": {
                                    "file_path": str(outside),
                                    "offset": 1,
                                    "limit": 20,
                                },
                            },
                            {
                                "type": "tool_use",
                                "name": "Read",
                                "input": {
                                    "file_path": str(source),
                                    "offset": 2,
                                    "limit": 2,
                                },
                            },
                        ]
                    }
                },
            },
        ),
        timed_out=False,
        metrics={},
    )

    context = _build_recovery_source_context(
        implementation,
        tmp_path,
        maximum_chars=1000,
    )

    assert "File: src/parser.py" in context
    assert "2: second = 2" in context
    assert "3: third = 3" in context
    assert "outside.py" not in context


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
        "host_test_attempted": True,
        "host_test_valid": False,
    }


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
    assert config.agent.visible_regression_test_timeout_seconds == 300
    assert config.agent.task_timeout_seconds == 1800
    assert config.agent.max_tool_output_chars == 12000


def test_scheduler_metrics_require_a_real_sandbox_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """只有沙箱返回结构化结果后，execution 和 passed 指标才能增加。"""

    request = StructuredTestPlanRequest(
        status="generated",
        target_argv=(
            "python", "-m", "pytest", "tests/test_one.py::test_bug"
        ),
        regression_argv=("python", "-m", "pytest", "tests/test_one.py"),
        origin="parent",
    )
    monkeypatch.setattr("scripts.run_claude.consume_test_plan", lambda _: request)

    class FakeSandbox:
        """记录 argv 并模拟一次成功的 Docker 测试。"""

        def run(
            self,
            repository,
            *,
            instance_id,
            base_commit,
            command,
            apply_patch,
        ):
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
    assert evidence.metrics()["visible_test_baseline_executions"] == 2
    assert evidence.metrics()["visible_test_baseline_passed"] == 2
    assert evidence.metrics()["visible_test_comparisons"] == 2
    assert evidence.metrics()["visible_test_new_regressions"] == 0
    assert evidence.metrics()["visible_test_rejected"] == 0
    assert evidence.metrics()["visible_test_parent_generated"] == 1
    assert evidence.ready_for_verification is True
    assert evidence.has_new_regression is False
    assert _verification_policy(evidence) == ("verification", None)


def test_scheduler_reuses_unchanged_successful_results_and_records_metrics(
    tmp_path: Path,
) -> None:
    """Final 阶段应复用稳定 baseline 和未变的成功 candidate，并把命中写入指标。"""

    request = StructuredTestPlanRequest(
        status="generated",
        target_argv=("python", "-m", "pytest", "tests/test_target.py"),
        regression_argv=("python", "-m", "pytest", "tests/test_neighbor.py"),
        origin="parent",
    )
    calls: list[tuple[tuple[str, ...], bool, float]] = []

    class FakeSandbox:
        """返回带耗时的成功结果，供测试确认第二阶段没有再次启动容器。"""

        timeout_seconds = 900

        def run(
            self,
            repository,
            *,
            instance_id,
            base_commit,
            command,
            apply_patch,
            timeout_seconds,
        ):
            calls.append((tuple(command), apply_patch, timeout_seconds))
            return VisibleTestResult(
                exit_code=0,
                output="passed\n",
                image="swebench/example:latest",
                timed_out=False,
                command_started=True,
                duration_seconds=2.5,
            )

    task = SWEbenchTask(
        instance_id="owner__repo-cache",
        repo="owner/repo",
        base_commit="b" * 40,
        problem_statement="Fix it.",
    )
    cache = {}
    arguments = {
        "task": task,
        "workspace_base_commit": "a" * 40,
        "sandbox": FakeSandbox(),
        "request": request,
        "candidate_patch": "diff --git a/a.py b/a.py\n-old\n+new\n",
        "cache": cache,
        "target_timeout_seconds": 900,
        "regression_timeout_seconds": 300,
    }

    initial = _execute_scheduled_test(tmp_path, **arguments)
    final = _execute_scheduled_test(tmp_path, **arguments)

    assert len(calls) == 4
    assert {timeout for _, _, timeout in calls} == {300, 900}
    assert initial.metrics()["duration_seconds"] == 10.0
    assert initial.metrics()["cache_hit"] is False
    assert final.metrics()["duration_seconds"] == 0.0
    assert final.metrics()["cache_hit"] is True
    assert final.metrics()["visible_test_executions"] == 0
    assert final.metrics()["visible_test_baseline_executions"] == 0
    assert all(
        result.cache_hit
        for execution in final.executions
        for result in (execution.baseline_result, execution.result)
    )


def test_scheduler_reruns_failed_candidate_but_reuses_baseline(
    tmp_path: Path,
) -> None:
    """候选失败不得被缓存掩盖，Final 仍需重跑 candidate 以容忍偶发故障。"""

    request = StructuredTestPlanRequest(
        status="generated",
        target_argv=("python", "-m", "pytest", "tests/test_target.py"),
        regression_argv=("python", "-m", "pytest", "tests/test_neighbor.py"),
        origin="parent",
    )
    calls: list[bool] = []

    class FakeSandbox:
        """所有 candidate 都失败，用调用次数证明它没有被复用。"""

        timeout_seconds = 900

        def run(self, repository, *, apply_patch, **kwargs):
            calls.append(apply_patch)
            return VisibleTestResult(
                exit_code=1 if apply_patch else 0,
                output="failed\n" if apply_patch else "passed\n",
                image="swebench/example:latest",
                timed_out=False,
                command_started=True,
                duration_seconds=1.0,
            )

    task = SWEbenchTask(
        instance_id="owner__repo-failed-cache",
        repo="owner/repo",
        base_commit="b" * 40,
        problem_statement="Fix it.",
    )
    cache = {}
    arguments = {
        "task": task,
        "workspace_base_commit": "a" * 40,
        "sandbox": FakeSandbox(),
        "request": request,
        "candidate_patch": "same patch",
        "cache": cache,
        "target_timeout_seconds": 900,
        "regression_timeout_seconds": 300,
    }

    _execute_scheduled_test(tmp_path, **arguments)
    final = _execute_scheduled_test(tmp_path, **arguments)

    assert calls.count(False) == 2
    assert calls.count(True) == 4
    assert all(execution.baseline_result.cache_hit for execution in final.executions)
    assert all(not execution.result.cache_hit for execution in final.executions)


def test_scheduler_records_exhausted_task_budget_without_starting_docker(
    tmp_path: Path,
) -> None:
    """总预算已耗尽时不得再启动容器，并应在事件和指标中明确记录原因。"""

    request = StructuredTestPlanRequest(
        status="generated",
        target_argv=("python", "-m", "pytest", "tests/test_target.py"),
        regression_argv=("python", "-m", "pytest", "tests/test_neighbor.py"),
        origin="parent",
    )

    class FakeSandbox:
        """预算门禁应在触达沙箱前结束调度。"""

        timeout_seconds = 900

        def run(self, *args, **kwargs):
            raise AssertionError("exhausted task budget must not start Docker")

    task = SWEbenchTask(
        instance_id="owner__repo-budget",
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
        candidate_patch="patch",
        cache={},
        target_timeout_seconds=900,
        regression_timeout_seconds=300,
        task_budget=_TaskBudget(deadline_monotonic=0.0),
    )
    phase = _scheduled_test_result(evidence)

    assert evidence.task_budget_exhausted is True
    assert evidence.metrics()["task_budget_exhausted"] is True
    assert phase.events[0]["details"]["task_budget_exhausted"] is True
    assert all(
        execution.baseline_result is None and execution.result is None
        for execution in evidence.executions
    )


def test_run_task_preserves_artifacts_when_total_budget_is_already_exhausted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """单题硬截止应生成 timeout 产物和五项指标，而不是抛异常丢失运行目录。"""

    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "widget.py"
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "widget.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    base_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    class FakeSandbox:
        """预算在 Agent 前耗尽，本测试只需要完成镜像元数据预检。"""

        def __init__(self, **kwargs):
            pass

        def resolve_image(self, instance_id):
            return "swebench/example:latest"

        def image_digest(self, image):
            return "sha256:test"

    monkeypatch.setattr("scripts.run_claude.VisibleTestSandbox", FakeSandbox)
    monkeypatch.setattr(
        "scripts.run_claude.RuntimeFingerprintCollector.collect",
        lambda self: {"schema_version": 1},
    )
    monkeypatch.setattr(
        _TaskBudget,
        "start",
        classmethod(lambda cls, timeout_seconds: cls(deadline_monotonic=0.0)),
    )
    monkeypatch.setattr(
        "scripts.run_claude._runner",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("expired budget must not start Claude Code")
        ),
    )
    task = SWEbenchTask(
        instance_id="owner__repo-budget-integration",
        repo="owner/repo",
        base_commit=base_commit,
        problem_statement="Fix it.",
    )

    run_path = run_claude_task(
        task,
        repository,
        ExperimentConfig.load(PROJECT_ROOT / "configs" / "dev_v2.yaml"),
        tmp_path / "runs",
        base_url="http://localhost:11434",
        workspace_base_commit=base_commit,
    )
    result = json.loads((run_path / "result.json").read_text(encoding="utf-8"))
    trajectory = json.loads(
        (run_path / "trajectory.json").read_text(encoding="utf-8")
    )

    assert result["run_status"] == "timeout"
    assert result["agent_exit_code"] == 124
    assert result["metrics"]["duration_seconds"] == 0.0
    assert result["metrics"]["cache_hit"] is False
    assert result["metrics"]["baseline_timed_out"] is False
    assert result["metrics"]["candidate_timed_out"] is False
    assert result["metrics"]["task_budget_exhausted"] is True
    assert trajectory["events"][-2]["event_type"] == "task_budget_exhausted"
    assert trajectory["events"][-1]["event_type"] == "patch_validation"


def test_scheduler_marks_only_baseline_pass_to_patched_fail_as_regression(
    tmp_path: Path,
) -> None:
    """基线已有失败不能误报成补丁回归，只有 pass→fail 才增加回归指标。"""

    request = StructuredTestPlanRequest(
        status="generated",
        target_argv=("python", "-m", "pytest", "tests/test_target.py"),
        regression_argv=("python", "-m", "pytest", "tests/test_neighbor.py"),
        origin="parent",
    )

    class FakeSandbox:
        """目标命令产生新失败，相邻命令在基线和 patch 上都保持失败。"""

        def run(
            self,
            repository,
            *,
            instance_id,
            base_commit,
            command,
            apply_patch,
        ):
            is_target = command == request.target_argv
            exit_code = 0 if is_target and not apply_patch else 1
            return VisibleTestResult(
                exit_code=exit_code,
                output="passed\n" if exit_code == 0 else "failed\n",
                image="swebench/example:latest",
                timed_out=False,
                command_started=True,
            )

    task = SWEbenchTask(
        instance_id="owner__repo-comparison",
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
    assert metrics["visible_test_comparisons"] == 2
    assert metrics["visible_test_new_regressions"] == 1
    assert metrics["visible_test_unchanged_baseline_failures"] == 1
    assert evidence.has_new_regression is True
    assert _verification_policy(evidence) == (
        "verification_regression_repair",
        ("Read", "Edit"),
    )
    assert "Comparison: new_regression" in evidence.prompt_text()
    assert "Comparison: baseline_failure_persists" in evidence.prompt_text()
    assert "Comparison: new_regression" in evidence.repair_prompt_text()
    assert "Comparison: baseline_failure_persists" not in evidence.repair_prompt_text()


def test_missing_plan_is_counted_and_blocks_verification() -> None:
    """缺失计划必须形成显式指标，但不再决定是否进入 Verification。"""

    evidence = ScheduledTestEvidence(
        request=StructuredTestPlanRequest(status="missing")
    )

    assert evidence.metrics()["visible_test_missing"] == 1
    assert evidence.metrics()["visible_test_executions"] == 0
    assert evidence.ready_for_verification is False


def test_nonempty_patch_enters_verification_without_test_plan() -> None:
    """测试计划缺失时，非空 patch 仍必须进入独立 Verification 会话。"""

    assert _should_run_verification("diff --git a/a.py b/a.py\n+new\n") is True
    assert _should_run_verification("\n\t") is False


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

        def run(
            self,
            repository,
            *,
            instance_id,
            base_commit,
            command,
            apply_patch,
        ):
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


def test_nonempty_patch_reaches_verification_when_parent_plan_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """集成保护：父进程找不到测试时，非空 patch 仍必须启动 Verification。"""

    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "src" / "widget.py"
    source.parent.mkdir()
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "src/widget.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    base_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    calls: list[str] = []

    class FakeAgent:
        """第一次产生 patch，第二次证明 Verification 实际启动。"""

        def run(self, worktree, prompt):
            phase = "verification" if "Current phase: verification" in prompt else "implementation"
            calls.append(phase)
            if phase == "implementation":
                source.write_text("value = 2\n", encoding="utf-8")
            return _result()

    class FakeSandbox:
        """本例没有可映射测试，因此 Docker run 不应被调用。"""

        def __init__(self, **kwargs):
            pass

        def resolve_image(self, instance_id):
            return "swebench/example:latest"

        def image_digest(self, image):
            return "sha256:test"

        def run(self, *args, **kwargs):
            raise AssertionError("missing parent plan must not invent a Docker command")

    monkeypatch.setattr("scripts.run_claude._runner", lambda *args, **kwargs: FakeAgent())
    monkeypatch.setattr("scripts.run_claude.VisibleTestSandbox", FakeSandbox)
    monkeypatch.setattr(
        "scripts.run_claude.RuntimeFingerprintCollector.collect",
        lambda self: {"schema_version": 1},
    )

    task = SWEbenchTask(
        instance_id="owner__project-1",
        repo="owner/project",
        base_commit=base_commit,
        problem_statement="Fix widget behavior.",
    )
    run_path = run_claude_task(
        task,
        repository,
        ExperimentConfig.load(PROJECT_ROOT / "configs" / "dev_v2.yaml"),
        tmp_path / "runs",
        base_url="http://localhost:11434",
        workspace_base_commit=base_commit,
    )

    result = json.loads((run_path / "result.json").read_text(encoding="utf-8"))
    assert calls == ["implementation", "verification"]
    assert "verification" in result["metrics"]["phases"]
    assert "test_protocol_gate" not in result["metrics"]["phases"]
    assert result["metrics"]["visible_test_missing"] == 2


def test_empty_patch_reuses_verification_budget_for_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """首轮空 patch 应启动可编辑的 Recovery，而不是浪费预留的模型 turns。"""

    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "src" / "widget.py"
    source.parent.mkdir()
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "src/widget.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    base_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    calls: list[str] = []
    runner_options: list[dict[str, object]] = []

    class FakeAgent:
        """首轮不修改文件，Recovery 会话交付一个既有源码修改。"""

        def run(self, worktree, prompt):
            if "empty-patch recovery implementation" in prompt:
                calls.append("recovery")
                source.write_text("value = 2\n", encoding="utf-8")
            elif "Current phase: verification" in prompt:
                calls.append("verification")
            else:
                calls.append("implementation")
            return _result()

    class FakeSandbox:
        """仓库没有测试文件，Recovery 后只应记录缺失计划。"""

        def __init__(self, **kwargs):
            pass

        def resolve_image(self, instance_id):
            return "swebench/example:latest"

        def image_digest(self, image):
            return "sha256:test"

        def run(self, *args, **kwargs):
            raise AssertionError("missing parent plan must not invent a Docker command")

    def fake_runner(*args, **kwargs):
        """记录各阶段 runner 选项，验证 Recovery 的 Bash 硬门禁。"""

        runner_options.append(dict(kwargs))
        return FakeAgent()

    monkeypatch.setattr("scripts.run_claude._runner", fake_runner)
    monkeypatch.setattr("scripts.run_claude.VisibleTestSandbox", FakeSandbox)
    monkeypatch.setattr(
        "scripts.run_claude.RuntimeFingerprintCollector.collect",
        lambda self: {"schema_version": 1},
    )

    task = SWEbenchTask(
        instance_id="owner__project-recovery",
        repo="owner/project",
        base_commit=base_commit,
        problem_statement="Fix widget behavior.",
    )
    run_path = run_claude_task(
        task,
        repository,
        ExperimentConfig.load(PROJECT_ROOT / "configs" / "dev_v2.yaml"),
        tmp_path / "runs",
        base_url="http://localhost:11434",
        workspace_base_commit=base_commit,
    )

    result = json.loads((run_path / "result.json").read_text(encoding="utf-8"))
    assert calls == ["implementation", "recovery"]
    assert "allow_bash" not in runner_options[0]
    assert runner_options[1]["allow_bash"] is False
    assert runner_options[1]["available_tools"] == ("Read", "Edit")
    assert runner_options[1]["turns"] == 2
    assert result["run_status"] == "completed"
    assert result["patch_generated"] is True
    assert "recovery_edit_gate" in result["metrics"]["phases"]
    assert "recovery_fallback" not in result["metrics"]["phases"]
    assert "scheduled_test_recovery" in result["metrics"]["phases"]
    assert "verification" not in result["metrics"]["phases"]
    assert result["metrics"]["visible_test_missing"] == 1


def test_recovery_fallback_uses_only_budget_left_after_edit_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """强制 Edit 未产出 patch 时，fallback 只能使用 Recovery 的剩余 turns。"""

    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "src" / "widget.py"
    source.parent.mkdir()
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "src/widget.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    base_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    calls: list[str] = []
    runner_options: list[dict[str, object]] = []

    class FakeAgent:
        """Edit gate 不落盘，fallback 使用剩余预算完成修改。"""

        def run(self, worktree, prompt):
            if "empty-patch recovery fallback" in prompt:
                calls.append("fallback")
                source.write_text("value = 2\n", encoding="utf-8")
            elif "mandatory Read-Edit gate" in prompt:
                calls.append("gate")
            else:
                calls.append("implementation")
            return _result()

    class FakeSandbox:
        """本例只验证阶段预算，不执行缺失的仓库测试。"""

        def __init__(self, **kwargs):
            pass

        def resolve_image(self, instance_id):
            return "swebench/example:latest"

        def image_digest(self, image):
            return "sha256:test"

        def run(self, *args, **kwargs):
            raise AssertionError("missing parent plan must not run Docker")

    def fake_runner(*args, **kwargs):
        """记录 gate/fallback 的 turns 与硬工具门禁。"""

        runner_options.append(dict(kwargs))
        return FakeAgent()

    monkeypatch.setattr("scripts.run_claude._runner", fake_runner)
    monkeypatch.setattr("scripts.run_claude.VisibleTestSandbox", FakeSandbox)
    monkeypatch.setattr(
        "scripts.run_claude.RuntimeFingerprintCollector.collect",
        lambda self: {"schema_version": 1},
    )
    task = SWEbenchTask(
        instance_id="owner__project-recovery-fallback",
        repo="owner/project",
        base_commit=base_commit,
        problem_statement="Fix widget behavior.",
    )

    run_path = run_claude_task(
        task,
        repository,
        ExperimentConfig.load(PROJECT_ROOT / "configs" / "dev_v2.yaml"),
        tmp_path / "runs",
        base_url="http://localhost:11434",
        workspace_base_commit=base_commit,
    )

    result = json.loads((run_path / "result.json").read_text(encoding="utf-8"))
    assert calls == ["implementation", "gate", "fallback"]
    assert runner_options[1]["turns"] == 2
    assert runner_options[2]["turns"] == 8
    assert runner_options[1]["allow_bash"] is False
    assert runner_options[2]["allow_bash"] is False
    assert runner_options[1]["available_tools"] == ("Read", "Edit")
    assert "available_tools" not in runner_options[2]
    assert "recovery_edit_gate" in result["metrics"]["phases"]
    assert "recovery_fallback" in result["metrics"]["phases"]
    assert result["patch_generated"] is True
