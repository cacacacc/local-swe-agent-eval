"""命令行入口：在无网络 SWE-bench instance 容器中运行仓库可见测试。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

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
    parser.add_argument(
        "--audit-path",
        type=Path,
        help="可选的 JSON 审计文件；任务专属短命令用它记录真实启动证据。",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _repository_changed(repository: Path, base_commit: str) -> bool:
    """判断前置测试启动前是否已存在相对任务基线的任何改动。

    同时检查已跟踪差异和未跟踪文件，因为模型可能通过 ``Edit``、``Write`` 或
    自行 commit 留下变化。只有两类检查都为空，才能证明测试发生在首次修改前。
    """

    tracked = subprocess.run(
        ["git", "-C", str(repository), "diff", "--quiet", base_commit, "--"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    untracked = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "ls-files",
            "--others",
            "--exclude-standard",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    # Git 查询异常时采取保守策略：不能把未知状态证明成“修改前测试”。
    return (
        tracked.returncode != 0
        or untracked.returncode != 0
        or bool(untracked.stdout.strip())
    )


def _append_audit(path: Path | None, attempt: dict[str, Any]) -> None:
    """原子追加一次 helper 尝试，避免非零测试退出时丢失执行证据。"""

    if path is None:
        return
    payload: dict[str, Any] = {"schema_version": 1, "attempts": []}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and isinstance(existing.get("attempts"), list):
            payload = existing
    payload["attempts"].append(attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _test_command_error(command: list[str]) -> str | None:
    """拒绝缺少测试 runner 的常见误拼接，同时保留仓库专用入口。

    上一轮模型多次只传 ``tests/test_x.py``，Docker 随后把源码当作 shell 程序。
    这里不试图枚举所有测试框架，只拦截空 argv、选项开头和裸 Python 测试路径；
    ``runtests.py`` 作为常见的可执行仓库入口仍被允许。
    """

    if not command:
        return "missing test command argv"
    first = command[0]
    if first.startswith("-"):
        return "test command must start with an executable, not an option"
    path_without_node_id = first.split("::", 1)[0]
    if path_without_node_id.endswith(".py") and Path(path_without_node_id).name not in {
        "runtests.py",
    }:
        return (
            "bare Python test path is not executable; use "
            "`python -m pytest <path>` or a repository test runner"
        )
    return None


def main() -> int:
    """执行测试并把截断后的真实输出及退出码透明返回给 Agent。"""

    arguments = build_parser().parse_args()
    command = list(arguments.command)
    if command and command[0] == "--":
        command.pop(0)
    repository = arguments.repository.resolve()
    attempt: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "repository_changed_before_test": _repository_changed(
            repository, arguments.base_commit
        ),
        "command_started": False,
        "exit_code": None,
        "timed_out": False,
        "image": None,
        "error": None,
    }
    command_error = _test_command_error(command)
    if command_error is not None:
        attempt["error"] = command_error
        _append_audit(arguments.audit_path, attempt)
        print(f"visible-test command error: {command_error}", file=sys.stderr)
        return 2
    try:
        result = VisibleTestSandbox(
            timeout_seconds=arguments.timeout,
            max_output_chars=arguments.max_output_chars,
        ).run(
            repository,
            instance_id=arguments.instance_id,
            base_commit=arguments.base_commit,
            command=command,
        )
    except (TestSandboxError, ValueError) as error:
        attempt["error"] = str(error)
        _append_audit(arguments.audit_path, attempt)
        print(f"visible-test sandbox error: {error}", file=sys.stderr)
        return 2
    attempt.update(
        {
            "command_started": result.command_started,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "image": result.image,
        }
    )
    _append_audit(arguments.audit_path, attempt)
    print(f"visible-test image: {result.image}")
    print(f"visible-test command-started: {str(result.command_started).lower()}")
    print(result.output, end="" if result.output.endswith("\n") else "\n")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
