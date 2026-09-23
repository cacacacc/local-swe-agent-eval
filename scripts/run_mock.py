"""通过 Phase 2 产物流水线运行一道确定性的 Mock 任务。

该 CLI 用于在不启动真实 LLM 的情况下端到端验证：任务加载、仓库准备、
Mock 修改、Git patch 收集以及运行产物持久化。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from agent.mock_runner import MockAgentRunner
from benchmark.repo_manager import RepositoryManager
from benchmark.swebench_loader import SWEbenchLoader
from benchmark.task import SWEbenchTask
from tracking.run_manager import RunManager


# 固定文本让每次 Mock 运行具有相同输入，便于比较产物。
MOCK_PROMPT = """Phase 2 mock run.
Create one deterministic marker file and emit only observable actions.
Do not access the network.
"""


def run_mock_task(
    task: SWEbenchTask,
    repository: Path | str,
    runs_root: Path | str,
) -> Path:
    """执行 Mock Agent，并返回已经完成写入的产物目录。"""

    session = RunManager(runs_root).start(
        task,
        phase="mock",
        agent="MockAgentRunner",
        model="mock-no-llm",
        prompt=MOCK_PROMPT,
    )
    try:
        result = MockAgentRunner().run(task, repository)
        patch = session.collect_patch(repository)
    except Exception as error:
        # Agent 失败时仍尽量收集部分 patch，避免丢失可用于诊断的现场。
        try:
            partial_patch = session.collect_patch(repository)
        except Exception as patch_error:
            partial_patch = ""
            patch_note = f"Patch collection also failed: {patch_error}\n"
        else:
            patch_note = ""
        session.finalize(
            exit_code=1,
            agent_log=(
                f"Mock agent failed with {type(error).__name__}: {error}\n"
                f"{patch_note}"
            ),
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
        events=[event.to_dict() for event in result.events],
        patch=patch,
    )
    return session.path


def _load_tasks(path: Path) -> SWEbenchLoader:
    """根据文件扩展名选择 JSON 或 JSONL 加载器。"""

    if path.suffix.lower() == ".jsonl":
        return SWEbenchLoader.from_jsonl(path)
    if path.suffix.lower() == ".json":
        return SWEbenchLoader.from_json(path)
    raise ValueError("--tasks must point to a .json or .jsonl file")


def build_parser() -> argparse.ArgumentParser:
    """声明命令行接口；路径默认值与实验配置的目录布局一致。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("repo-cache"))
    parser.add_argument("--workspace-root", type=Path, default=Path("workspaces"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow repository clone/fetch during environment preparation.",
    )
    return parser


def main() -> int:
    """解析参数、准备仓库、运行 Mock 流水线并打印产物路径。"""

    arguments = build_parser().parse_args()
    task = _load_tasks(arguments.tasks).get(arguments.instance_id)
    prepared = RepositoryManager(
        arguments.cache_root,
        arguments.workspace_root,
    ).prepare(task, allow_network=arguments.allow_network)
    run_path = run_mock_task(task, prepared.path, arguments.runs_root)
    print(run_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
