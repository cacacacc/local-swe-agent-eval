"""使用本地 Ollama 驱动 Claude Code，完成一道已准备的 SWE-bench 任务。

输入任务文件必须是经过 ``SWEbenchLoader`` 安全投影的 JSON/JSONL，不能直接把
包含 gold patch 或 test patch 的原始数据记录传给本脚本。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re

from agent.claude_runner import ClaudeCodeRunner
from agent.prompt_builder import PromptBuilder
from benchmark.repo_manager import RepositoryManager
from benchmark.swebench_loader import SWEbenchLoader
from benchmark.task import SWEbenchTask
from experiment.config import ExperimentConfig
from experiment.runtime_fingerprint import RuntimeFingerprintCollector
from tracking.run_manager import RunManager


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


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
    session = RunManager(runs_root).start(
        task,
        phase=config.experiment.phase,
        agent=config.agent.framework,
        model=config.model.name,
        prompt=prompt,
        configuration=configuration,
    )
    runner = ClaudeCodeRunner(
        model=config.model.name,
        timeout_seconds=config.agent.timeout_seconds,
        max_turns=config.agent.max_turns,
        context_length=config.model.context_length,
        max_output_tokens=config.model.max_output_tokens,
        base_url=base_url,
    )

    try:
        result = runner.run(repository, prompt)
        patch = session.collect_patch(repository)
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
