"""命令行入口：在无网络 SWE-bench instance 容器中运行仓库可见测试。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 该脚本会被目标 worktree 中的 Claude Code 通过绝对路径调用；显式加入项目根目录，
# 避免当前目录不是调度仓库时无法导入 ``agent`` 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.test_sandbox import TestSandboxError, VisibleTestSandbox  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """声明固定任务身份、资源上限以及 ``--`` 后的无 shell 测试 argv。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-output-chars", type=int, default=12000)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    """执行测试并把截断后的真实输出及退出码透明返回给 Agent。"""

    arguments = build_parser().parse_args()
    command = list(arguments.command)
    if command and command[0] == "--":
        command.pop(0)
    try:
        result = VisibleTestSandbox(
            timeout_seconds=arguments.timeout,
            max_output_chars=arguments.max_output_chars,
        ).run(
            arguments.repository,
            instance_id=arguments.instance_id,
            base_commit=arguments.base_commit,
            command=command,
        )
    except (TestSandboxError, ValueError) as error:
        print(f"visible-test sandbox error: {error}", file=sys.stderr)
        return 2
    print(f"visible-test image: {result.image}")
    print(result.output, end="" if result.output.endswith("\n") else "\n")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
