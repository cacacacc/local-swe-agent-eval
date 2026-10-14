"""Verify that Implementation Bash searches cannot execute host tests or shell code."""

from pathlib import Path
import subprocess

import pytest

from agent.host_command_guard import HostCommandDenied, guarded_argv


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("rg -n 'edge case' src", ["rg", "-n", "edge case", "src"]),
        ("git status --short", ["git", "-c", "core.fsmonitor=false"]),
        ("git diff -- src/widget.py", ["git", "-c", "core.fsmonitor=false"]),
        ("head -n 20 src/widget.py", ["head", "-n", "20", "src/widget.py"]),
    ],
)
def test_guard_allows_single_read_only_search(
    tmp_path: Path, command: str, expected: list[str],
) -> None:
    """Common exploration commands remain available as direct argv without shell evaluation."""

    argv = guarded_argv(command, tmp_path)
    assert argv[: len(expected)] == expected


@pytest.mark.parametrize(
    "command",
    [
        "python .agent-issue-repro.py",
        "python3 -m pytest tests/test_widget.py",
        "pytest -q",
        "bash -c 'python .agent-issue-repro.py'",
        "rg --pre 'python .agent-issue-repro.py' edge .",
        "rg edge . | head -n 20",
        "rg edge . && python .agent-issue-repro.py",
        "git diff --output=/tmp/leak",
        "git -c alias.diff='!python .agent-issue-repro.py' diff",
        "cat ../outside.py",
        "head -n 10 /etc/passwd",
    ],
)
def test_guard_denies_execution_or_workspace_escape(tmp_path: Path, command: str) -> None:
    """Model-controlled shell syntax, test runners, and executable options fail closed."""

    with pytest.raises(HostCommandDenied):
        guarded_argv(command, tmp_path)


def test_guard_process_rejects_python_before_it_can_run(tmp_path: Path) -> None:
    """The real prefix executable returns 126 without running a proposed host script."""

    marker = tmp_path / "ran.txt"
    guard = Path(__file__).resolve().parents[1] / "agent" / "host_command_guard.py"
    command = f"python -c 'from pathlib import Path; Path({str(marker)!r}).touch()'"
    result = subprocess.run(
        [str(guard), command],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 126
    assert "Host command denied" in result.stderr
    assert not marker.exists()


def test_guard_process_runs_a_read_only_search(tmp_path: Path) -> None:
    """The prefix remains useful for Implementation exploration, not just denial."""

    (tmp_path / "widget.py").write_text("boundary_value = 3\n", encoding="utf-8")
    guard = Path(__file__).resolve().parents[1] / "agent" / "host_command_guard.py"
    result = subprocess.run(
        [str(guard), "rg -n boundary_value widget.py"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "1:boundary_value = 3" in result.stdout


def test_guard_accepts_claude_code_shell_wrapper_without_executing_it(tmp_path: Path) -> None:
    """The real Claude prefix protocol must extract pwd and ignore its shell setup."""

    guard = Path(__file__).resolve().parents[1] / "agent" / "host_command_guard.py"
    wrapped = (
        "source /home/user/.claude/shell-snapshots/snapshot-bash-1.sh 2>/dev/null "
        "|| true && { shopt -u extglob || setopt NO_EXTENDED_GLOB NO_BARE_GLOB_QUAL; } "
        ">/dev/null 2>&1 || true && { \\builtin unalias -- 'unsetenv'; "
        "\\builtin unset -f -- 'unsetenv'; } >/dev/null 2>&1 || true "
        "&& eval pwd < /dev/null && pwd -P >| /tmp/claude-abc123-cwd"
    )
    result = subprocess.run(
        [str(guard), wrapped],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == str(tmp_path)


def test_guard_accepts_quoted_search_inside_claude_wrapper(tmp_path: Path) -> None:
    """CLI quoting around a multi-argument search must preserve the search arguments."""

    (tmp_path / "widget.py").write_text("boundary_value = 3\n", encoding="utf-8")
    guard = Path(__file__).resolve().parents[1] / "agent" / "host_command_guard.py"
    wrapped = (
        "source /tmp/snapshot.sh 2>/dev/null || true && eval "
        "'rg -n boundary_value widget.py' < /dev/null && pwd -P "
        ">| /tmp/claude-abc123-cwd"
    )
    result = subprocess.run(
        [str(guard), wrapped],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "1:boundary_value = 3" in result.stdout


def test_guard_rejects_claude_wrapped_host_test(tmp_path: Path) -> None:
    """Unwrapping CLI boilerplate must not allow Python or test runners on the host."""

    guard = Path(__file__).resolve().parents[1] / "agent" / "host_command_guard.py"
    wrapped = (
        "source /tmp/snapshot.sh 2>/dev/null || true && eval 'python3 -m pytest -q' "
        "< /dev/null && pwd -P >| /tmp/claude-abc123-cwd"
    )
    result = subprocess.run(
        [str(guard), wrapped],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 126
    assert "host execution is not allowed: python3" in result.stderr


def test_guard_does_not_expand_shell_substitutions_inside_search_text(tmp_path: Path) -> None:
    """A quoted search pattern remains argv text rather than invoking a subshell."""

    marker = tmp_path / "substitution_ran"
    guard = Path(__file__).resolve().parents[1] / "agent" / "host_command_guard.py"
    result = subprocess.run(
        [str(guard), f'rg "$(touch {marker})" .'],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert not marker.exists()
