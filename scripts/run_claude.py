"""使用本地 Ollama 驱动 Claude Code，完成一道已准备的 SWE-bench 任务。

输入任务文件必须是经过 ``SWEbenchLoader`` 安全投影的 JSON/JSONL，不能直接把
包含 gold patch 或 test patch 的原始数据记录传给本脚本。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import re
import shlex
import sys

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
from agent.test_sandbox import VisibleTestSandbox
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
) -> ClaudeCodeRunner:
    """按阶段预算构造隔离的 Claude Code 进程。"""

    return ClaudeCodeRunner(
        model=config.model.name,
        timeout_seconds=timeout_seconds,
        max_turns=turns,
        context_length=config.model.context_length,
        max_output_tokens=config.model.max_output_tokens,
        base_url=base_url,
    )


def _apply_patch_gate(result: ClaudeCodeResult, patch: str) -> ClaudeCodeResult:
    """把空补丁从“正常完成”改为明确失败，并记录测试证据是否存在。

    该门禁不把“运行过测试”当作成功，因为部分仓库在宿主环境没有依赖；真正的
    resolved 状态仍只由官方 harness 决定。它只消除模型通用回复造成的假完成。
    """

    patch_generated = bool(patch.strip())
    metrics = dict(result.metrics)
    metrics["patch_gate"] = {
        "patch_generated": patch_generated,
        "test_attempted": bool(result.test_output.strip()),
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


def _visible_test_command(task: SWEbenchTask, config: ExperimentConfig) -> str | None:
    """生成写入 Prompt 的固定沙箱前缀；正式基线默认不启用。"""

    if not config.agent.visible_test_sandbox:
        return None
    script = config.project_root / "scripts" / "run_visible_tests.py"
    if not script.is_file():
        raise FileNotFoundError(f"visible test runner does not exist: {script}")
    return shlex.join(
        [
            sys.executable,
            str(script),
            "--repository",
            ".",
            "--instance-id",
            task.instance_id,
            "--base-commit",
            task.base_commit,
            "--timeout",
            str(config.agent.visible_test_timeout_seconds),
            "--max-output-chars",
            str(config.agent.max_tool_output_chars),
            "--",
        ]
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
) -> Path:
    """执行真实 Agent，失败或超时时也尽量保存现场与部分 patch。"""

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
            visible_test_command = _visible_test_command(task, config)
            implementation_prompt = build_implementation_phase_prompt(
                prompt,
                implementation_turns=implementation_turns,
                verification_turns=config.agent.verification_turns,
                max_file_read_lines=config.agent.max_file_read_lines,
                max_tool_output_chars=config.agent.max_tool_output_chars,
                visible_test_command=visible_test_command,
            )
            verification_prompt = build_verification_phase_prompt(
                prompt,
                verification_turns=config.agent.verification_turns,
                max_file_read_lines=config.agent.max_file_read_lines,
                max_tool_output_chars=config.agent.max_tool_output_chars,
                visible_test_command=visible_test_command,
            )
            implementation_result = _runner(
                config,
                turns=implementation_turns,
                timeout_seconds=implementation_timeout,
                base_url=base_url,
            ).run(repository, implementation_prompt)
            # 第二会话从干净上下文开始，但直接看到第一阶段留在 worktree 的 diff；
            # 任务正文会被重新注入，因此不依赖易失败的 auto-compact 摘要。
            verification_result = _runner(
                config,
                turns=config.agent.verification_turns,
                timeout_seconds=verification_timeout,
                base_url=base_url,
            ).run(repository, verification_prompt)
            result = combine_phase_results(
                (
                    ("implementation", implementation_result),
                    ("verification", verification_result),
                )
            )
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
    run_path = run_claude_task(
        task,
        prepared.path,
        config,
        config.project_root / config.storage.runs / arguments.run_id,
        base_url=arguments.base_url,
    )
    print(run_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
