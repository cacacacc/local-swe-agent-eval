"""Command-line entry point: run repository-visible tests in a network-isolated SWE-bench instance container."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# This script is invoked via absolute path by Claude Code inside the target worktree;
# explicitly add the project root so the ``agent`` package can still be imported when
# the current directory is not the scheduling repository.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.test_sandbox import TestSandboxError, VisibleTestSandbox  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Declare the fixed task identity, resource limits, and the no-shell test argv after ``--``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-output-chars", type=int, default=12000)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    """Run the test and transparently return the truncated real output and exit code to the Agent."""

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
