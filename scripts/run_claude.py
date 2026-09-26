"""使用本地 Ollama 驱动 Claude Code，完成一道已准备的 SWE-bench 任务。

输入任务文件必须是经过 ``SWEbenchLoader`` 安全投影的 JSON/JSONL，不能直接把
包含 gold patch 或 test patch 的原始数据记录传给本脚本。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

from agent.claude_runner import (
    ClaudeCodeResult,
    ClaudeCodeRunner,
    combine_phase_results,
)
from agent.prompt_builder import (
    PromptBuilder,
    build_implementation_phase_prompt,
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


def _runner(
    config: ExperimentConfig,
    *,
    turns: int,
    timeout_seconds: int,
    base_url: str,
    allow_bash: bool = True,
) -> ClaudeCodeRunner:
    """按阶段预算构造 Claude Code；验证阶段可在 CLI 层禁用 Bash。"""

    return ClaudeCodeRunner(
        model=config.model.name,
        timeout_seconds=timeout_seconds,
        max_turns=turns,
        context_length=config.model.context_length,
        max_output_tokens=config.model.max_output_tokens,
        base_url=base_url,
        allow_bash=allow_bash,
    )


def _apply_patch_gate(result: ClaudeCodeResult, patch: str) -> ClaudeCodeResult:
    """把空补丁从“正常完成”改为明确失败，并记录测试证据是否存在。

    该门禁不把“运行过测试”当作成功，因为部分仓库在宿主环境没有依赖；真正的
    resolved 状态仍只由官方 harness 决定。它只消除模型通用回复造成的假完成。
    """

    patch_generated = bool(patch.strip())
    metrics = dict(result.metrics)
    visible_executions = int(metrics.get("visible_test_executions", 0))
    host_attempts = int(metrics.get("host_test_calls", 0))
    metrics["patch_gate"] = {
        "patch_generated": patch_generated,
        "test_attempted": visible_executions > 0 or host_attempts > 0,
        "visible_test_attempted": visible_executions > 0,
        "host_test_attempted": host_attempts > 0,
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
    if not patch_generated and exit_code == 0:
        # 2 表示运行器的交付物门禁失败，区别于 Claude CLI 自身的退出码 1。
        exit_code = 2
    return replace(result, exit_code=exit_code, events=tuple(events), metrics=metrics)


@dataclass(frozen=True, slots=True)
class ScheduledTestEvidence:
    """一次调度器测试尝试及其可审计结果。"""

    request: TestPlanRequest
    result: VisibleTestResult | None = None
    error: str | None = None

    def metrics(self) -> dict[str, int | bool]:
        """生成不会把文本提及误算为容器执行的指标。"""

        executed = self.result is not None
        return {
            "agent_turns": 0,
            "tool_calls": 0,
            "host_test_calls": 0,
            "agent_test_command_calls": 0,
            "visible_test_calls": int(executed),
            "visible_test_requests": int(self.request.requested),
            "visible_test_rejected": int(
                self.request.status == "rejected" or self.error is not None
            ),
            "visible_test_executions": int(executed),
            "visible_test_passed": int(executed and self.result.exit_code == 0),
            "visible_test_timed_out": bool(executed and self.result.timed_out),
            "timed_out": False,
            "token_usage": {},
        }

    def prompt_text(self) -> str:
        """把真实执行证据压缩成可直接注入修复会话的文本。"""

        if self.request.status == "missing":
            return "No structured test plan was submitted by the implementation session."
        if self.request.status == "rejected":
            return f"The submitted test plan was rejected: {self.request.error}"
        command = " ".join(self.request.argv)
        if self.error is not None:
            return f"The scheduled test `{command}` was rejected: {self.error}"
        assert self.result is not None
        return (
            f"Scheduled visible test argv: {list(self.request.argv)!r}\n"
            f"Docker image: {self.result.image}\n"
            f"Exit code: {self.result.exit_code}\n"
            f"Timed out: {self.result.timed_out}\n"
            "Output:\n"
            f"{self.result.output.rstrip()}"
        )


def _execute_scheduled_test(
    repository: Path,
    *,
    task: SWEbenchTask,
    workspace_base_commit: str,
    sandbox: VisibleTestSandbox,
    fallback_argv: tuple[str, ...] = (),
) -> ScheduledTestEvidence:
    """消费模型计划并在 Docker 中执行；缺省时可复测上一轮 argv。

    无效计划只形成 rejected 证据而不终止整个任务，保证修复会话仍能看到明确
    原因。若修复会话没有提交新计划，则复用首次已验证的 argv，避免把重复写控制
    文件浪费成模型 turns。
    """

    request = consume_test_plan(repository)
    if request.status == "missing" and fallback_argv:
        request = TestPlanRequest(status="accepted", argv=fallback_argv)
    if not request.accepted:
        return ScheduledTestEvidence(request=request)
    try:
        result = sandbox.run(
            repository,
            instance_id=task.instance_id,
            base_commit=workspace_base_commit,
            command=request.argv,
        )
    except (TestSandboxError, ValueError) as error:
        return ScheduledTestEvidence(request=request, error=str(error))
    return ScheduledTestEvidence(request=request, result=result)


def _scheduled_test_result(evidence: ScheduledTestEvidence) -> ClaudeCodeResult:
    """把调度器证据包装成 phase，使 trajectory、日志和汇总保持同一拓扑。"""

    details: dict[str, Any] = {
        "request_status": evidence.request.status,
        "argv": list(evidence.request.argv),
        "error": evidence.request.error or evidence.error,
    }
    if evidence.result is not None:
        details.update(
            {
                "exit_code": evidence.result.exit_code,
                "image": evidence.result.image,
                "timed_out": evidence.result.timed_out,
                "output": evidence.result.output,
            }
        )
    text = evidence.prompt_text()
    return ClaudeCodeResult(
        # 调度测试失败是修复输入而非 Claude 进程失败；总体 exit code 仍由验证会话
        # 决定，官方 resolved 则继续只由 harness 决定。
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
                config.agent.max_turns - config.agent.verification_turns
            )
            # 墙钟时间按 turns 同比例硬切分，确保实现阶段即使卡住，也不能侵占
            # 独立验证会话的修复时间。
            implementation_timeout = max(
                1,
                config.agent.timeout_seconds
                * implementation_turns
                // config.agent.max_turns,
            )
            verification_timeout = max(
                1,
                config.agent.timeout_seconds - implementation_timeout,
            )
            implementation_prompt = build_implementation_phase_prompt(
                prompt,
                implementation_turns=implementation_turns,
                verification_turns=config.agent.verification_turns,
                max_file_read_lines=config.agent.max_file_read_lines,
                max_tool_output_chars=config.agent.max_tool_output_chars,
            )
            implementation_result = _runner(
                config,
                turns=implementation_turns,
                timeout_seconds=implementation_timeout,
                base_url=base_url,
            ).run(repository, implementation_prompt)
            if config.agent.visible_test_sandbox:
                initial_evidence = _execute_scheduled_test(
                    repository,
                    task=task,
                    workspace_base_commit=patch_base_commit,
                    sandbox=test_sandbox,
                )
            else:
                initial_evidence = ScheduledTestEvidence(
                    request=TestPlanRequest(status="missing")
                )
            # 测试计划已经被消费并删除，此时收集的 candidate patch 不会夹带调度
            # 控制文件。显式比较隔离仓库基线也能覆盖模型擅自创建的 commit。
            candidate_patch = session.collect_patch(repository)
            candidate_patch_for_prompt = truncate_output(
                candidate_patch,
                config.agent.max_tool_output_chars,
            )
            verification_prompt = build_verification_phase_prompt(
                prompt,
                verification_turns=config.agent.verification_turns,
                max_file_read_lines=config.agent.max_file_read_lines,
                max_tool_output_chars=config.agent.max_tool_output_chars,
                candidate_patch=candidate_patch_for_prompt,
                scheduled_test_evidence=initial_evidence.prompt_text(),
            )
            # 第二会话从干净上下文开始，但直接看到第一阶段留在 worktree 的 diff；
            # 任务正文会被重新注入，因此不依赖易失败的 auto-compact 摘要。
            verification_result = _runner(
                config,
                turns=config.agent.verification_turns,
                timeout_seconds=verification_timeout,
                base_url=base_url,
                allow_bash=False,
            ).run(repository, verification_prompt)
            if config.agent.visible_test_sandbox:
                fallback_argv = (
                    initial_evidence.request.argv
                    if initial_evidence.request.accepted
                    else ()
                )
                final_evidence = _execute_scheduled_test(
                    repository,
                    task=task,
                    workspace_base_commit=patch_base_commit,
                    sandbox=test_sandbox,
                    fallback_argv=fallback_argv,
                )
            else:
                final_evidence = ScheduledTestEvidence(
                    request=TestPlanRequest(status="missing")
                )
            result = combine_phase_results(
                (
                    ("implementation", implementation_result),
                    (
                        "scheduled_test_initial",
                        _scheduled_test_result(initial_evidence),
                    ),
                    ("verification", verification_result),
                    (
                        "scheduled_test_final",
                        _scheduled_test_result(final_evidence),
                    ),
                )
            )
            # combine 默认采用最后一个 phase 的退出码；最后一阶段是调度器测试，
            # 因此显式恢复 Claude 验证会话状态，避免测试失败被误报成进程异常。
            result = replace(result, exit_code=verification_result.exit_code)
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
