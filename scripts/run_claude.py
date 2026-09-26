"""使用本地 Ollama 驱动 Claude Code，完成一道已准备的 SWE-bench 任务。

输入任务文件必须是经过 ``SWEbenchLoader`` 安全投影的 JSON/JSONL，不能直接把
包含 gold patch 或 test patch 的原始数据记录传给本脚本。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import sys
from typing import Any

from agent.claude_runner import (
    ClaudeCodeResult,
    ClaudeCodeRunner,
    combine_phase_results,
)
from agent.prompt_builder import (
    PromptBuilder,
    build_implementation_phase_prompt,
    build_test_planning_phase_prompt,
    build_verification_phase_prompt,
)
from agent.test_plan import TestPlanRequest, consume_test_plan
from agent.test_sandbox import (
    TestSandboxError,
    VisibleTestResult,
    VisibleTestSandbox,
    truncate_output,
)
from benchmark.repo_manager import RepositoryManager
from benchmark.swebench_loader import SWEbenchLoader
from benchmark.task import SWEbenchTask
from experiment.config import ExperimentConfig
from experiment.runtime_fingerprint import RuntimeFingerprintCollector
from tracking.run_manager import RunManager


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".jinja",
    ".jinja2",
    ".js",
    ".jsx",
    ".php",
    ".pxd",
    ".py",
    ".pyi",
    ".pyx",
    ".rb",
    ".rs",
    ".scss",
    ".ts",
    ".tsx",
}


@dataclass(frozen=True, slots=True)
class BaselineTestLauncher:
    """一条任务专属短命令及其独立审计文件。

    launcher 位于运行产物目录而非待修复仓库中，因此不会污染候选 patch，也不会
    被 Docker 沙箱误当成待应用的代码变化。instance、commit 和资源上限全部在
    创建时绑定，模型只负责提供正常的测试 argv。
    """

    command: str
    path: Path
    audit_path: Path


def _create_baseline_test_launcher(
    run_path: Path,
    *,
    repository: Path,
    task: SWEbenchTask,
    base_commit: str,
    config: ExperimentConfig,
) -> BaselineTestLauncher:
    """创建可直接执行的 ``visible-test``，隐藏易抄错的长参数。"""

    launcher_path = run_path / "visible-test"
    audit_path = run_path / "baseline-test-audit.json"
    bound_command = shlex.join(
        (
            sys.executable,
            str(config.project_root / "scripts" / "run_visible_tests.py"),
            "--repository",
            str(repository.resolve()),
            "--instance-id",
            task.instance_id,
            "--base-commit",
            base_commit,
            "--timeout",
            str(config.agent.visible_test_timeout_seconds),
            "--max-output-chars",
            str(config.agent.max_tool_output_chars),
            "--audit-path",
            str(audit_path),
            "--",
        )
    )
    # 使用极小的 POSIX shell 转发器，``"$@"`` 保证 pytest node id 等参数不会
    # 被二次拆词；其目录只加入本题进程 PATH，避免并发任务互相串用 launcher。
    launcher_path.write_text(
        "#!/bin/sh\n"
        "if [ \"$#\" -eq 0 ]; then\n"
        "  echo 'usage: visible-test <test executable> [args ...]' >&2\n"
        "  exit 2\n"
        "fi\n"
        f"exec {bound_command} \"$@\"\n",
        encoding="utf-8",
    )
    launcher_path.chmod(0o700)
    return BaselineTestLauncher(
        command=launcher_path.name,
        path=launcher_path,
        audit_path=audit_path,
    )


def _attach_baseline_test_evidence(
    result: ClaudeCodeResult,
    audit_path: Path,
) -> ClaudeCodeResult:
    """把 helper 的落盘事实加入 Implementation 指标与轨迹。

    非零测试退出码仍表示测试真实启动；它常常正是 bug 的基线复现。有效前置测试
    只要求恰好调用一次、Docker 命令已启动且调用前仓库未发生变化。
    """

    attempts: list[dict[str, Any]] = []
    if audit_path.is_file():
        try:
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("attempts"), list):
            attempts = [item for item in payload["attempts"] if isinstance(item, dict)]

    executions = sum(item.get("command_started") is True for item in attempts)
    before_edit_executions = sum(
        item.get("command_started") is True
        and item.get("repository_changed_before_test") is False
        for item in attempts
    )
    passed = sum(
        item.get("command_started") is True and item.get("exit_code") == 0
        for item in attempts
    )
    valid = len(attempts) == 1 and before_edit_executions == 1
    metrics = dict(result.metrics)
    metrics.update(
        {
            "implementation_baseline_test_attempts": len(attempts),
            "implementation_baseline_test_executions": executions,
            "implementation_baseline_test_before_edit_executions": (
                before_edit_executions
            ),
            "implementation_baseline_test_passed": passed,
            "implementation_baseline_test_valid": valid,
        }
    )
    events = list(result.events)
    events.append(
        {
            "sequence": len(events) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "implementation_baseline_test_evidence",
            "details": {
                "attempts": attempts,
                "attempt_count": len(attempts),
                "execution_count": executions,
                "before_edit_execution_count": before_edit_executions,
                "passed_count": passed,
                "valid": valid,
            },
        }
    )
    return replace(result, events=tuple(events), metrics=metrics)


def _runner(
    config: ExperimentConfig,
    *,
    turns: int,
    timeout_seconds: int,
    base_url: str,
    allow_bash: bool = True,
    tool_path: Path | None = None,
) -> ClaudeCodeRunner:
    """按阶段预算构造 Claude Code，并可注入该题专属工具目录。"""

    return ClaudeCodeRunner(
        model=config.model.name,
        timeout_seconds=timeout_seconds,
        max_turns=turns,
        context_length=config.model.context_length,
        max_output_tokens=config.model.max_output_tokens,
        base_url=base_url,
        allow_bash=allow_bash,
        tool_path=tool_path,
    )


def _apply_patch_gate(result: ClaudeCodeResult, patch: str) -> ClaudeCodeResult:
    """把空补丁从“正常完成”改为明确失败，并记录测试证据是否存在。

    该门禁不把“运行过测试”当作成功，因为部分仓库在宿主环境没有依赖；真正的
    resolved 状态仍只由官方 harness 决定。它只消除模型通用回复造成的假完成。
    """

    patch_generated = bool(patch.strip())
    existing_source_modified = _patch_modifies_existing_source(patch)
    metrics = dict(result.metrics)
    visible_executions = int(metrics.get("visible_test_executions", 0))
    host_attempts = int(metrics.get("host_test_calls", 0))
    implementation_baseline_attempts = int(
        metrics.get("implementation_baseline_test_attempts", 0)
    )
    implementation_baseline_executions = int(
        metrics.get("implementation_baseline_test_executions", 0)
    )
    implementation_baseline_valid = bool(
        metrics.get("implementation_baseline_test_valid", False)
    )
    metrics["patch_gate"] = {
        "patch_generated": patch_generated,
        "existing_source_modified": existing_source_modified,
        # 宿主测试没有使用 SWE-bench instance 环境，永远不能满足测试门禁。
        "test_attempted": visible_executions > 0 or implementation_baseline_valid,
        "visible_test_attempted": visible_executions > 0,
        "implementation_baseline_test_attempted": implementation_baseline_attempts > 0,
        "implementation_baseline_test_executed": (
            implementation_baseline_executions > 0
        ),
        "implementation_baseline_test_valid": implementation_baseline_valid,
        "host_test_attempted": host_attempts > 0,
        "host_test_valid": False,
    }
    events = list(result.events)
    events.append(
        {
            "sequence": len(events) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "patch_validation",
            "details": metrics["patch_gate"],
        }
    )
    exit_code = result.exit_code
    if (not patch_generated or not existing_source_modified) and exit_code == 0:
        # 2 表示运行器的交付物门禁失败，区别于 Claude CLI 自身的退出码 1。
        exit_code = 2
    return replace(result, exit_code=exit_code, events=tuple(events), metrics=metrics)


def _patch_modifies_existing_source(patch: str) -> bool:
    """判断补丁是否修改至少一个既有的非测试源码文件。

    该门禁专门拦截只留下复现脚本、临时文本或新增测试目录的探索结果。已有测试
    文件的修改也不算产品修复，避免模型通过改测试绕过交付要求。
    """

    for section in re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE):
        match = re.match(r"diff --git a/(.+?) b/(.+?)\n", section)
        if match is None or "new file mode " in section:
            continue
        path = Path(match.group(2))
        lowered_parts = {part.lower() for part in path.parts}
        name = path.name.lower()
        is_test = bool(lowered_parts & {"test", "tests", "testing"}) or (
            name.startswith("test_")
            or name.endswith("_test.py")
            or name.endswith(".test.js")
            or name.endswith(".test.ts")
        )
        # 文档、日志和任意既有临时文件同样不能满足“修复产品源码”的要求；
        # 显式后缀白名单适配 SWE-bench 中常见的 Python/C/前端与模板源码。
        if not is_test and path.suffix.lower() in _SOURCE_SUFFIXES and "@@" in section:
            return True
    return False


def _protocol_failure_result(message: str) -> ClaudeCodeResult:
    """构造阻止 verification 的明确协议失败，而不丢失已有候选补丁。"""

    return ClaudeCodeResult(
        exit_code=3,
        agent_log=message.rstrip() + "\n",
        test_output=message.rstrip() + "\n",
        events=(
            {
                "sequence": 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "test_protocol_failure",
                "details": {"message": message},
            },
        ),
        timed_out=False,
        metrics={
            "agent_turns": 0,
            "tool_calls": 0,
            "host_test_calls": 0,
            "agent_test_command_calls": 0,
            "visible_test_calls": 0,
            "visible_test_requests": 0,
            "visible_test_missing": 0,
            "visible_test_rejected": 0,
            "visible_test_executions": 0,
            "visible_test_passed": 0,
            "visible_test_timed_out": False,
            "timed_out": False,
            "token_usage": {},
        },
    )


@dataclass(frozen=True, slots=True)
class ScheduledTestExecution:
    """测试计划中一条带用途标签的真实 Docker 执行结果。"""

    label: str
    argv: tuple[str, ...]
    result: VisibleTestResult | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ScheduledTestEvidence:
    """一份双命令测试计划及其可审计 Docker 执行结果。"""

    request: TestPlanRequest
    executions: tuple[ScheduledTestExecution, ...] = ()

    @property
    def ready_for_verification(self) -> bool:
        """仅当目标与回归命令都真实进入 Docker 后允许验证会话启动。"""

        return (
            self.request.accepted
            and len(self.executions) == len(self.request.commands) == 2
            and all(
                execution.result is not None and execution.result.command_started
                for execution in self.executions
            )
        )

    def metrics(self) -> dict[str, int | bool]:
        """生成不会把文本提及误算为容器执行的指标。"""

        executed = sum(
            item.result is not None and item.result.command_started
            for item in self.executions
        )
        rejected = self.request.status == "rejected" or any(
            item.error is not None
            or (item.result is not None and not item.result.command_started)
            for item in self.executions
        )
        return {
            "agent_turns": 0,
            "tool_calls": 0,
            "host_test_calls": 0,
            "agent_test_command_calls": 0,
            "visible_test_calls": executed,
            "visible_test_requests": int(self.request.requested),
            "visible_test_missing": int(self.request.status == "missing"),
            "visible_test_rejected": int(rejected),
            "visible_test_executions": executed,
            "visible_test_passed": sum(
                item.result is not None and item.result.exit_code == 0
                and item.result.command_started
                for item in self.executions
            ),
            "visible_test_timed_out": any(
                item.result is not None
                and item.result.command_started
                and item.result.timed_out
                for item in self.executions
            ),
            "timed_out": False,
            "token_usage": {},
        }

    def prompt_text(self) -> str:
        """把真实执行证据压缩成可直接注入修复会话的文本。"""

        if self.request.status == "missing":
            return "No structured test plan was submitted by the implementation session."
        if self.request.status == "rejected":
            return f"The submitted test plan was rejected: {self.request.error}"
        blocks: list[str] = []
        for execution in self.executions:
            command = " ".join(execution.argv)
            if execution.error is not None:
                blocks.append(
                    f"[{execution.label}] `{command}` could not run: {execution.error}"
                )
                continue
            assert execution.result is not None
            blocks.append(
                f"[{execution.label}] argv: {list(execution.argv)!r}\n"
                f"Docker image: {execution.result.image}\n"
                f"Command started: {execution.result.command_started}\n"
                f"Exit code: {execution.result.exit_code}\n"
                f"Timed out: {execution.result.timed_out}\n"
                "Output:\n"
                f"{execution.result.output.rstrip()}"
            )
        return "\n\n".join(blocks)


def _execute_scheduled_test(
    repository: Path,
    *,
    task: SWEbenchTask,
    workspace_base_commit: str,
    sandbox: VisibleTestSandbox,
    request: TestPlanRequest | None = None,
) -> ScheduledTestEvidence:
    """消费或复用双命令计划，并分别在一次性 Docker 容器中执行。"""

    request = consume_test_plan(repository) if request is None else request
    if not request.accepted:
        return ScheduledTestEvidence(request=request)
    executions: list[ScheduledTestExecution] = []
    for label, argv in request.commands:
        try:
            result = sandbox.run(
                repository,
                instance_id=task.instance_id,
                base_commit=workspace_base_commit,
                command=argv,
            )
        except (TestSandboxError, ValueError) as error:
            executions.append(
                ScheduledTestExecution(label=label, argv=argv, error=str(error))
            )
        else:
            executions.append(
                ScheduledTestExecution(label=label, argv=argv, result=result)
            )
    return ScheduledTestEvidence(request=request, executions=tuple(executions))


def _scheduled_test_result(evidence: ScheduledTestEvidence) -> ClaudeCodeResult:
    """把调度器证据包装成 phase，使 trajectory、日志和汇总保持同一拓扑。"""

    details: dict[str, Any] = {
        "request_status": evidence.request.status,
        "commands": [
            {
                "label": execution.label,
                "argv": list(execution.argv),
                "error": execution.error,
                "exit_code": (
                    execution.result.exit_code
                    if execution.result is not None
                    else None
                ),
                "image": (
                    execution.result.image if execution.result is not None else None
                ),
                "timed_out": (
                    execution.result.timed_out
                    if execution.result is not None
                    else False
                ),
                "command_started": (
                    execution.result.command_started
                    if execution.result is not None
                    else False
                ),
            }
            for execution in evidence.executions
        ],
        "error": evidence.request.error,
    }
    text = evidence.prompt_text()
    return ClaudeCodeResult(
        # 此 phase 只负责保存证据；是否允许进入 verification 由父进程随后依据
        # ready_for_verification 判定，不能用这里的零退出码绕过真实执行门禁。
        exit_code=0,
        agent_log=text + "\n",
        test_output=text + "\n",
        events=(
            {
                "sequence": 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "visible_test_execution",
                "details": details,
            },
        ),
        timed_out=False,
        metrics=evidence.metrics(),
    )


def prepare_visible_test_image(
    task: SWEbenchTask,
    config: ExperimentConfig,
    *,
    allow_network_preparation: bool,
) -> str | None:
    """在 Agent 启动前准备测试镜像；求解阶段不得调用此下载入口。"""

    if not config.agent.visible_test_sandbox:
        return None
    sandbox = VisibleTestSandbox(
        timeout_seconds=config.agent.visible_test_timeout_seconds,
        max_output_chars=config.agent.max_tool_output_chars,
    )
    return sandbox.ensure_image(
        task.instance_id,
        allow_pull=allow_network_preparation,
    )


def _load_tasks(path: Path) -> SWEbenchLoader:
    """按扩展名加载经过字段白名单约束的任务快照。"""

    if path.suffix.lower() == ".jsonl":
        return SWEbenchLoader.from_jsonl(path)
    if path.suffix.lower() == ".json":
        return SWEbenchLoader.from_json(path)
    raise ValueError("--tasks must point to a .json or .jsonl file")


def run_claude_task(
    task: SWEbenchTask,
    repository: Path,
    config: ExperimentConfig,
    runs_root: Path,
    *,
    base_url: str,
    workspace_base_commit: str | None = None,
) -> Path:
    """执行实现、调度测试与无 Bash 修复阶段，并持久化完整证据。"""

    patch_base_commit = workspace_base_commit or task.base_commit
    prompt = PromptBuilder.from_file(config.prompt_template_path).build(task)
    runtime_fingerprint = RuntimeFingerprintCollector(
        project_root=config.project_root,
        prompt_path=config.prompt_template_path,
        model_name=config.model.name,
        base_url=base_url,
    ).collect()
    configuration = config.to_metadata()
    configuration["runtime"] = runtime_fingerprint
    if config.agent.visible_test_sandbox:
        # 在创建 Agent 进程前完成只读镜像预检；缺失镜像必须回到允许联网的环境
        # 准备阶段处理，不能让模型在正式求解期间尝试 docker pull。
        test_sandbox = VisibleTestSandbox(
            timeout_seconds=config.agent.visible_test_timeout_seconds,
            max_output_chars=config.agent.max_tool_output_chars,
        )
        visible_test_image = test_sandbox.resolve_image(task.instance_id)
        configuration["visible_test_runtime"] = {
            "image": visible_test_image,
            "image_id": test_sandbox.image_digest(visible_test_image),
            "network": "none",
        }
    session = RunManager(runs_root).start(
        task,
        phase=config.experiment.phase,
        agent=config.agent.framework,
        model=config.model.name,
        prompt=prompt,
        configuration=configuration,
        patch_base_commit=patch_base_commit,
    )
    try:
        if config.agent.verification_turns:
            implementation_turns = (
                config.agent.max_turns
                - config.agent.verification_turns
                - config.agent.test_planning_turns
            )
            # 三个会话的墙钟预算按 turns 比例硬切分，防止实现探索侵占测试规划
            # 和验证修复时间；最后一段吸收整数除法余数。
            implementation_timeout = max(
                1,
                config.agent.timeout_seconds
                * implementation_turns
                // config.agent.max_turns,
            )
            planning_timeout = max(
                1,
                config.agent.timeout_seconds
                * config.agent.test_planning_turns
                // config.agent.max_turns,
            )
            verification_timeout = max(
                1,
                config.agent.timeout_seconds
                - implementation_timeout
                - planning_timeout,
            )
            baseline_launcher = _create_baseline_test_launcher(
                session.path,
                repository=repository,
                task=task,
                base_commit=patch_base_commit,
                config=config,
            )
            implementation_prompt = build_implementation_phase_prompt(
                prompt,
                implementation_turns=implementation_turns,
                verification_turns=config.agent.verification_turns,
                max_file_read_lines=config.agent.max_file_read_lines,
                max_tool_output_chars=config.agent.max_tool_output_chars,
                baseline_test_command=baseline_launcher.command,
            )
            implementation_result = _runner(
                config,
                turns=implementation_turns,
                timeout_seconds=implementation_timeout,
                base_url=base_url,
                tool_path=baseline_launcher.path.parent,
            ).run(repository, implementation_prompt)
            implementation_result = _attach_baseline_test_evidence(
                implementation_result,
                baseline_launcher.audit_path,
            )
            phases: list[tuple[str, ClaudeCodeResult]] = [
                ("implementation", implementation_result)
            ]

            # 先消费实现会话的计划，再收集候选补丁，确保一次性控制文件永远不会
            # 混入交付物。缺失或拒绝时必须进入独立 planner，不能静默跳到验证。
            initial_request = consume_test_plan(repository)
            candidate_patch = session.collect_patch(repository)
            candidate_patch_for_prompt = truncate_output(
                candidate_patch,
                config.agent.max_tool_output_chars,
            )

            if initial_request.accepted:
                accepted_request = initial_request
            else:
                phases.append(
                    (
                        "test_plan_submission",
                        _scheduled_test_result(
                            ScheduledTestEvidence(request=initial_request)
                        ),
                    )
                )
                previous_error = (
                    "missing"
                    if initial_request.status == "missing"
                    else initial_request.error or "rejected"
                )
                planning_prompt = build_test_planning_phase_prompt(
                    prompt,
                    planning_turns=config.agent.test_planning_turns,
                    max_file_read_lines=config.agent.max_file_read_lines,
                    candidate_patch=candidate_patch_for_prompt,
                    previous_plan_error=previous_error,
                )
                planning_result = _runner(
                    config,
                    turns=config.agent.test_planning_turns,
                    timeout_seconds=planning_timeout,
                    base_url=base_url,
                    allow_bash=False,
                ).run(repository, planning_prompt)
                phases.append(("test_planning", planning_result))
                accepted_request = consume_test_plan(repository)
                # Planner 的唯一授权写入是已被消费的控制文件。任何产品或测试源码
                # 变化都会污染候选补丁，因此按协议失败处理。
                if session.collect_patch(repository) != candidate_patch:
                    accepted_request = TestPlanRequest(
                        status="rejected",
                        error="test planner modified the candidate patch",
                    )

            initial_evidence = _execute_scheduled_test(
                repository,
                task=task,
                workspace_base_commit=patch_base_commit,
                sandbox=test_sandbox,
                request=accepted_request,
            )
            phases.append(
                ("scheduled_test_initial", _scheduled_test_result(initial_evidence))
            )
            if not initial_evidence.ready_for_verification:
                failure = _protocol_failure_result(
                    "Verification was not started because both target and regression "
                    "tests did not execute in Docker."
                )
                phases.append(("test_protocol_gate", failure))
                result = combine_phase_results(tuple(phases))
                result = replace(result, exit_code=failure.exit_code)
            else:
                verification_prompt = build_verification_phase_prompt(
                    prompt,
                    verification_turns=config.agent.verification_turns,
                    max_file_read_lines=config.agent.max_file_read_lines,
                    max_tool_output_chars=config.agent.max_tool_output_chars,
                    candidate_patch=candidate_patch_for_prompt,
                    scheduled_test_evidence=truncate_output(
                        initial_evidence.prompt_text(),
                        config.agent.max_tool_output_chars,
                    ),
                )
                # 验证会话只有在两类测试都真实进入 Docker 后才会启动，并直接看到
                # 退出码与输出；Bash 在 CLI 层禁用，避免回退到宿主环境测试。
                verification_result = _runner(
                    config,
                    turns=config.agent.verification_turns,
                    timeout_seconds=verification_timeout,
                    base_url=base_url,
                    allow_bash=False,
                ).run(repository, verification_prompt)
                phases.append(("verification", verification_result))

                replacement_request = consume_test_plan(repository)
                final_request = (
                    initial_evidence.request
                    if replacement_request.status == "missing"
                    else replacement_request
                )
                final_evidence = _execute_scheduled_test(
                    repository,
                    task=task,
                    workspace_base_commit=patch_base_commit,
                    sandbox=test_sandbox,
                    request=final_request,
                )
                phases.append(
                    ("scheduled_test_final", _scheduled_test_result(final_evidence))
                )
                result = combine_phase_results(tuple(phases))
                if final_evidence.ready_for_verification:
                    # Docker 测试的非零退出码属于最终证据，不覆盖 Claude 会话的运行
                    # 状态；官方 resolved 仍只由 SWE-bench harness 决定。
                    result = replace(result, exit_code=verification_result.exit_code)
                else:
                    failure = _protocol_failure_result(
                        "Final target and regression tests did not both execute in Docker."
                    )
                    phases.append(("test_protocol_gate_final", failure))
                    result = combine_phase_results(tuple(phases))
                    result = replace(result, exit_code=failure.exit_code)
        else:
            # 已冻结的正式基线配置继续走原来的单会话路径，保证历史 fingerprint
            # 对应的实验协议不被后续架构开发悄悄改写。
            result = _runner(
                config,
                turns=config.agent.max_turns,
                timeout_seconds=config.agent.timeout_seconds,
                base_url=base_url,
            ).run(repository, prompt)
        patch = session.collect_patch(repository)
        result = _apply_patch_gate(result, patch)
    except Exception as error:
        # 无论 Agent 还是 patch 收集失败，都要终结 metadata 的 running 状态，
        # 否则后续分析无法区分“仍在运行”和“基础设施异常退出”。
        try:
            partial_patch = session.collect_patch(repository)
        except Exception as patch_error:
            partial_patch = ""
            patch_note = f"; patch collection failed: {patch_error}"
        else:
            patch_note = ""
        session.finalize(
            exit_code=1,
            agent_log=f"{type(error).__name__}: {error}{patch_note}\n",
            test_output="",
            events=[
                {
                    "sequence": 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "agent_error",
                    "details": {
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                }
            ],
            patch=partial_patch,
        )
        raise

    session.finalize(
        exit_code=result.exit_code,
        agent_log=result.agent_log,
        test_output=result.test_output,
        events=result.events,
        patch=patch,
        metrics=result.metrics,
    )
    return session.path


def build_parser() -> argparse.ArgumentParser:
    """声明单题运行参数；run ID 强制显式给出以防覆盖或结果误复用。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument(
        "--allow-network-preparation",
        action="store_true",
        help="Allow Git clone/fetch before the offline Agent subprocess starts.",
    )
    return parser


def main() -> int:
    """校验配置、准备 worktree，并启动只连接本机 Ollama 的 Agent。"""

    arguments = build_parser().parse_args()
    if not _SAFE_RUN_ID.fullmatch(arguments.run_id):
        raise ValueError("--run-id may contain only letters, digits, dot, underscore and dash")

    config = ExperimentConfig.load(arguments.config)
    if config.experiment.phase == "evaluation":
        config.require_frozen()
    task = _load_tasks(arguments.tasks).get(arguments.instance_id)
    manager = RepositoryManager(
        config.project_root / config.storage.repository_cache,
        config.project_root / config.storage.workspaces / arguments.run_id,
    )
    prepared = manager.prepare(
        task,
        allow_network=arguments.allow_network_preparation,
    )
    prepare_visible_test_image(
        task,
        config,
        allow_network_preparation=arguments.allow_network_preparation,
    )
    run_path = run_claude_task(
        task,
        prepared.path,
        config,
        config.project_root / config.storage.runs / arguments.run_id,
        base_url=arguments.base_url,
        workspace_base_commit=prepared.workspace_base_commit,
    )
    print(run_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
