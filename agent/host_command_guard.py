#!/usr/bin/python3
"""Run only audited read-only Bash-tool commands without invoking a shell.

Claude Code passes a shell-setup wrapper containing the Bash command as one
argument to ``CLAUDE_CODE_SHELL_PREFIX``. This guard extracts the command,
parses it into an argv, and executes a small allowlist directly;
shell substitutions, pipes, redirects, Python, and test runners never execute.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import sys


class HostCommandDenied(ValueError):
    """A model-proposed host command is outside the read-only search policy."""


_RG_FLAGS = {
    "-n", "--line-number", "-i", "--ignore-case", "-S", "--smart-case",
    "-F", "--fixed-strings", "-l", "--files-with-matches", "-c", "--count",
    "--files", "--hidden", "--no-ignore", "--no-heading", "--color=never",
}
_RG_VALUE_FLAGS = {"-g", "--glob", "-m", "--max-count", "-C", "--context", "-A", "-B"}
_RG_NUMERIC_FLAGS = {"-m", "--max-count", "-C", "--context", "-A", "-B"}
_GIT_DIFF_FLAGS = {
    "--stat", "--name-only", "--name-status", "--numstat", "--check",
    "--cached", "--no-ext-diff", "--no-textconv", "-p", "--patch",
}
_GIT_STATUS_FLAGS = {"--short", "--porcelain", "--porcelain=v1", "--branch"}
_CLAUDE_CWD_SUFFIX = re.compile(
    r"(?s)(.*) < /dev/null && pwd -P >\| /tmp/claude-[A-Za-z0-9-]+-cwd\Z"
)


def _unwrap_claude_command(value: str) -> str:
    """Extract the model command from Claude Code's shell setup without running it.

    Claude Code 2.1.280 passes a snapshot prologue, ``eval`` invocation, and
    working-directory redirect as one prefix argument. Executing that wrapper
    would bypass the guard, so only its quoted command text is retained.
    """

    if " && eval " not in value:
        return value
    if len(value) > 32_768 or not value.startswith("source "):
        raise HostCommandDenied("unexpected Claude Bash wrapper")
    _, _, wrapped_command = value.partition(" && eval ")
    match = _CLAUDE_CWD_SUFFIX.fullmatch(wrapped_command)
    if match is None:
        raise HostCommandDenied("unexpected Claude Bash wrapper")
    try:
        words = shlex.split(match.group(1), posix=True)
    except ValueError as error:
        raise HostCommandDenied("invalid Claude Bash quoting") from error
    if not words:
        raise HostCommandDenied("empty Claude Bash command")
    # The CLI passes either one shell-quoted command string or several simple
    # eval words. Joining their literal values never performs shell expansion.
    return " ".join(words)


def _inside_workspace(value: str, workspace: Path) -> bool:
    """Reject absolute and traversing paths before a read-only command opens them."""

    path = Path(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and (workspace / path).resolve().is_relative_to(workspace)
    )


def _rg_argv(arguments: list[str], workspace: Path) -> list[str]:
    """Accept common bounded ripgrep searches while excluding executable preprocessors."""

    positional: list[str] = []
    index = 0
    files_mode = False
    while index < len(arguments):
        value = arguments[index]
        if value == "--":
            positional.extend(arguments[index + 1 :])
            break
        if value in _RG_FLAGS:
            files_mode |= value == "--files"
        elif value in _RG_VALUE_FLAGS:
            index += 1
            if index >= len(arguments):
                raise HostCommandDenied(f"missing value for {value}")
            if value in _RG_NUMERIC_FLAGS and not arguments[index].isdigit():
                raise HostCommandDenied(f"{value} requires a numeric value")
            if value in {"-g", "--glob"} and arguments[index].startswith("-"):
                raise HostCommandDenied("rg glob value cannot be another option")
        elif value.startswith(("--glob=", "--max-count=", "--context=")):
            if value.startswith("--max-count=") and not value.removeprefix("--max-count=").isdigit():
                raise HostCommandDenied("rg max-count requires a numeric value")
            if value.startswith("--context=") and not value.removeprefix("--context=").isdigit():
                raise HostCommandDenied("rg context requires a numeric value")
        elif value.startswith(("-g", "-m", "-C", "-A", "-B")) and len(value) > 2:
            if value[:2] != "-g" and not value[2:].isdigit():
                raise HostCommandDenied(f"unsupported rg option: {value}")
        elif value.startswith("-"):
            raise HostCommandDenied(f"unsupported rg option: {value}")
        else:
            positional.append(value)
        index += 1
    paths = positional if files_mode else positional[1:]
    if not files_mode and not positional:
        raise HostCommandDenied("rg requires a pattern")
    if not all(_inside_workspace(path, workspace) for path in paths):
        raise HostCommandDenied("rg paths must stay inside the task repository")
    return ["rg", *arguments]


def _file_view_argv(name: str, arguments: list[str], workspace: Path) -> list[str]:
    """Accept only plain path reads and simple output-size options."""

    if name == "pwd":
        if arguments:
            raise HostCommandDenied("pwd takes no arguments")
        return ["pwd"]
    if name == "ls":
        flags = {"-l", "-a", "-la", "-al", "-1", "-A"}
    elif name in {"head", "tail"}:
        flags = {"-n", "-c"}
    elif name == "wc":
        flags = {"-l", "-c"}
    else:
        flags = set()
    paths: list[str] = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value in flags:
            if name in {"head", "tail"}:
                index += 1
                if index >= len(arguments) or not arguments[index].isdigit():
                    raise HostCommandDenied(f"{name} requires a numeric limit")
        elif name in {"head", "tail"} and value.startswith("-n") and value[2:].isdigit():
            pass
        elif value.startswith("-"):
            raise HostCommandDenied(f"unsupported {name} option: {value}")
        else:
            paths.append(value)
        index += 1
    if name in {"cat", "head", "tail", "wc"} and not paths:
        raise HostCommandDenied(f"{name} requires a repository file")
    if not all(_inside_workspace(path, workspace) for path in paths):
        raise HostCommandDenied("file paths must stay inside the task repository")
    return [name, *arguments]


def _git_argv(arguments: list[str], workspace: Path) -> list[str]:
    """Disable Git's external diff and fsmonitor escape hatches for status/diff."""

    if not arguments or arguments[0] not in {"status", "diff"}:
        raise HostCommandDenied("only git status and git diff are allowed")
    subcommand = arguments[0]
    flags = _GIT_STATUS_FLAGS if subcommand == "status" else _GIT_DIFF_FLAGS
    paths_started = False
    for value in arguments[1:]:
        if value == "--" and not paths_started:
            paths_started = True
        elif paths_started:
            if not _inside_workspace(value, workspace):
                raise HostCommandDenied("git path must stay inside the task repository")
        elif value not in flags and not (
            subcommand == "diff" and
            (value.startswith("--unified=") and value.removeprefix("--unified=").isdigit()
             or value.startswith("-U") and value[2:].isdigit())
        ):
            raise HostCommandDenied(f"unsupported git argument: {value}")
    safe_diff_flags = ["--no-ext-diff", "--no-textconv"] if subcommand == "diff" else []
    return [
        "git", "-c", "core.fsmonitor=false", "-c", "diff.external=",
        subcommand, *safe_diff_flags, *arguments[1:],
    ]


def guarded_argv(command: str, workspace: Path | str) -> list[str]:
    """Convert one model Bash string into a safe argv, or fail closed before execution."""

    if not command or len(command) > 16_384 or "\n" in command or "\x00" in command:
        raise HostCommandDenied("empty, multiline, or oversized host command")
    try:
        words = shlex.split(command, posix=True)
    except ValueError as error:
        raise HostCommandDenied("invalid shell quoting") from error
    if not words:
        raise HostCommandDenied("empty host command")
    if any(word in {"|", "||", "&&", ";", "&", ">", ">>", "<", "2>"} for word in words):
        raise HostCommandDenied("shell operators are not allowed")
    root = Path(workspace).resolve()
    executable, *arguments = words
    if executable == "rg":
        return _rg_argv(arguments, root)
    if executable in {"pwd", "ls", "cat", "head", "tail", "wc"}:
        return _file_view_argv(executable, arguments, root)
    if executable == "git":
        return _git_argv(arguments, root)
    raise HostCommandDenied(f"host execution is not allowed: {executable}")


def main() -> int:
    """Run the allowed command directly; never pass untrusted text to a shell."""

    if len(sys.argv) != 2:
        print("Host command denied: expected one Bash command", file=sys.stderr)
        return 126
    try:
        argv = guarded_argv(_unwrap_claude_command(sys.argv[1]), Path.cwd())
    except HostCommandDenied as error:
        print(f"Host command denied: {error}. Parent Docker owns tests.", file=sys.stderr)
        return 126
    # A model cannot pass shell assignments through guarded_argv, but inherited
    # tool-specific configuration can still add executable helpers (rg --pre,
    # Git fsmonitor/external diff). Clear those environment escape hatches too.
    os.environ.pop("RIPGREP_CONFIG_PATH", None)
    for name in tuple(os.environ):
        if name in {
            "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS", "GIT_EXEC_PATH",
            "GIT_EXTERNAL_DIFF", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM",
        } or name.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            os.environ.pop(name, None)
    os.environ["GIT_CONFIG_NOSYSTEM"] = "1"
    os.environ["GIT_CONFIG_GLOBAL"] = os.devnull
    os.environ["GIT_PAGER"] = "/bin/cat"
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    # Resolve only from trusted system tool directories, never a task-controlled
    # PATH entry that might contain an executable named rg or git.
    binary = shutil.which(
        argv[0], path="/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"
    )
    if binary is None:
        print(f"Host command unavailable: {argv[0]}", file=sys.stderr)
        return 127
    try:
        os.execv(binary, argv)
    except OSError as error:
        print(f"Host command unavailable: {error}", file=sys.stderr)
        return 127
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
