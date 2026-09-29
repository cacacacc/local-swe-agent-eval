"""Verify Claude Code command boundaries, outbound network restrictions, trajectory cleaning, and test output extraction."""

import json

import pytest

from agent.claude_runner import (
    ClaudeCodeError,
    ClaudeCodeResult,
    ClaudeCodeRunner,
    combine_phase_results,
)


def make_runner() -> ClaudeCodeRunner:
    """Build a runner that matches the key limits of the official experiment but does not actually start a process."""

    return ClaudeCodeRunner(
        model="qwen3.5:9b",
        timeout_seconds=5400,
        max_turns=30,
        context_length=32768,
        max_output_tokens=8192,
    )


def test_command_fixes_model_turn_limit_and_noninteractive_isolation() -> None:
    """The command must pin the local model and turn limit, and isolate user plugins and session history."""

    command = make_runner().command("Solve the task")

    assert command[0] == "claude"
    assert command[command.index("--print") + 1] == "Solve the task"
    assert command[command.index("--model") + 1] == "qwen3.5:9b"
    assert command[command.index("--max-turns") + 1] == "30"
    assert "--bare" in command
    assert "--no-session-persistence" in command
    assert "WebFetch" in command and "WebSearch" in command
    # The prompt must appear before the variable-length tool list so the CLI does not swallow it as a rule.
    assert command.index("Solve the task") < command.index("--disallowed-tools")


def test_environment_removes_cloud_credentials_and_keeps_only_local_endpoint() -> None:
    """The subprocess must not inherit cloud credentials, and common network clients may only reach the local machine by bypassing proxies."""

    environment = make_runner().environment(
        {
            "PATH": "/usr/bin",
            "ANTHROPIC_API_KEY": "must-not-leak",
            "AWS_SECRET_ACCESS_KEY": "must-not-leak",
            "HTTP_PROXY": "http://inherited.invalid:8080",
            "ALL_PROXY": "socks5://inherited.invalid:1080",
        }
    )

    assert "ANTHROPIC_API_KEY" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert environment["ANTHROPIC_AUTH_TOKEN"] == "ollama"
    assert environment["ANTHROPIC_BASE_URL"] == "http://localhost:11434"
    assert environment["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "32768"
    assert environment["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "8192"
    assert "HTTP_PROXY" not in environment
    assert "ALL_PROXY" not in environment
    assert environment["HTTPS_PROXY"] == "http://127.0.0.1:9"
    assert environment["NO_PROXY"] == "localhost,127.0.0.1,::1"


def test_runner_rejects_remote_model_endpoint() -> None:
    """On misconfiguration, remote model services must be rejected before the Agent starts."""

    with pytest.raises(ClaudeCodeError, match="local HTTP"):
        ClaudeCodeRunner(
            model="qwen3.5:9b",
            timeout_seconds=10,
            max_turns=1,
            context_length=4096,
            max_output_tokens=1024,
            base_url="https://api.anthropic.com",
        )


def test_stream_parser_removes_thinking_and_extracts_test_output() -> None:
    """The trajectory must not store hidden reasoning, but must associate test commands with their tool results."""

    tool_call = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "thinking", "thinking": "private analysis"},
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "Bash",
                    "input": {"command": "python -m pytest -q"},
                },
            ]
        },
    }
    tool_result = {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "content": "2 passed",
                }
            ]
        },
    }
    stdout = "\n".join(json.dumps(item) for item in (tool_call, tool_result))

    events, agent_log, test_output = ClaudeCodeRunner._parse_stream(stdout, "")

    assert len(events) == 2
    assert "private analysis" not in agent_log
    assert '"thinking"' not in agent_log
    assert "$ python -m pytest -q" in test_output
    assert "2 passed" in test_output


def test_summary_counts_turns_tools_and_final_usage_without_double_counting() -> None:
    """Reported metrics must come from structured events and use the cumulative token usage of the final result."""

    events = [
        {
            "event_type": "assistant",
            "details": {
                "message": {
                    "content": [
                        {"type": "text", "text": "Inspecting."},
                        {"type": "tool_use", "id": "1", "name": "Read"},
                    ]
                }
            },
        },
        {
            "event_type": "result",
            "details": {"usage": {"input_tokens": 120, "output_tokens": 30}},
        },
    ]

    metrics = ClaudeCodeRunner._summarize(events, timed_out=False)

    assert metrics == {
        "agent_turns": 1,
        "tool_calls": 1,
        "visible_test_calls": 0,
        "visible_test_requests": 0,
        "visible_test_missing": 0,
        "visible_test_rejected": 0,
        "visible_test_parent_generated": 0,
        "visible_test_executions": 0,
        "visible_test_valid_executions": 0,
        "visible_test_infrastructure_errors": 0,
        "visible_test_baseline_failure_signatures": 0,
        "visible_test_candidate_failure_signatures": 0,
        "visible_test_new_failure_signatures": 0,
        "visible_test_outcome_parser_failures": 0,
        "test_evidence_available": False,
        "visible_test_passed": 0,
        "visible_test_baseline_executions": 0,
        "visible_test_baseline_passed": 0,
        "visible_test_comparisons": 0,
        "visible_test_new_regressions": 0,
        "visible_test_fixed_baseline_failures": 0,
        "visible_test_unchanged_baseline_failures": 0,
        "visible_test_timed_out": False,
        "duration_seconds": 0.0,
        "cache_hit": False,
        "baseline_timed_out": False,
        "candidate_timed_out": False,
        "task_budget_exhausted": False,
        "agent_test_command_calls": 0,
        "host_test_calls": 0,
        "timed_out": False,
        "token_usage": {"input_tokens": 120, "output_tokens": 30},
    }


def test_summary_preserves_observable_terminal_reason() -> None:
    """The termination reason must enter structured metrics to avoid relying on truncated log text during analysis."""

    metrics = ClaudeCodeRunner._summarize(
        [
            {
                "event_type": "result",
                "details": {
                    "terminal_reason": "max_turns",
                    "subtype": "error_max_turns",
                    "num_turns": 30,
                    "usage": {"input_tokens": 10},
                },
            }
        ],
        timed_out=False,
    )

    assert metrics["terminal_reason"] == "max_turns"
    assert metrics["result_subtype"] == "error_max_turns"
    assert metrics["model_turns"] == 30


def test_combine_phase_results_uses_verification_exit_and_sums_metrics() -> None:
    """After the implementation phase is exhausted, an independent verification phase can still repair the task, and the audit metrics of both phases must be preserved."""

    implementation = ClaudeCodeResult(
        exit_code=1,
        agent_log="implementation log\n",
        test_output="",
        events=({"sequence": 1, "event_type": "result", "details": {}},),
        timed_out=False,
        metrics={
            "agent_turns": 30,
            "tool_calls": 12,
            "timed_out": False,
            "token_usage": {"input_tokens": 100, "output_tokens": 20},
            "terminal_reason": "max_turns",
        },
    )
    verification = ClaudeCodeResult(
        exit_code=0,
        agent_log="verification log\n",
        test_output="$ pytest\n1 passed\n",
        events=({"sequence": 1, "event_type": "result", "details": {}},),
        timed_out=False,
        metrics={
            "agent_turns": 5,
            "tool_calls": 3,
            "timed_out": False,
            "token_usage": {"input_tokens": 40, "output_tokens": 10},
        },
    )

    combined = combine_phase_results(
        (("implementation", implementation), ("verification", verification))
    )

    assert combined.exit_code == 0
    assert combined.metrics["agent_turns"] == 35
    assert combined.metrics["tool_calls"] == 15
    assert combined.metrics["visible_test_calls"] == 0
    assert combined.metrics["visible_test_executions"] == 0
    assert combined.metrics["visible_test_parent_generated"] == 0
    assert combined.metrics["host_test_calls"] == 0
    assert combined.metrics["token_usage"] == {
        "input_tokens": 140,
        "output_tokens": 30,
    }
    assert combined.metrics["phases"]["implementation"]["terminal_reason"] == "max_turns"
    assert [event["phase"] for event in combined.events] == [
        "implementation",
        "implementation",
        "verification",
        "verification",
    ]


def test_verification_runner_disables_bash_at_cli_boundary() -> None:
    """The verification session must disable Bash at the CLI level, not merely rely on the model following the prompt."""

    runner = ClaudeCodeRunner(
        model="qwen3.5:9b",
        timeout_seconds=60,
        max_turns=10,
        context_length=32768,
        max_output_tokens=8192,
        allow_bash=False,
    )

    command = runner.command("Verify")

    assert "Bash" in command[command.index("--disallowed-tools") + 1 :]


def test_runner_can_expose_only_one_tool_for_each_recovery_gate_step() -> None:
    """Each CLI call in the Recovery state machine may only expose the tools required for the current step."""

    runner = ClaudeCodeRunner(
        model="qwen3.5:9b",
        timeout_seconds=60,
        max_turns=1,
        context_length=32768,
        max_output_tokens=8192,
        allow_bash=False,
        available_tools=("Read",),
    )

    command = runner.command("Edit now")
    available = command[command.index("--tools") + 1]

    assert available == "Read"
    assert "Bash" in command[command.index("--disallowed-tools") + 1 :]


def test_runner_uses_explicit_persistent_session_for_read_edit_state_machine() -> None:
    """The Edit step must resume an explicit UUID, not attach to an arbitrary recent session in the directory."""

    session_id = "12345678-1234-5678-1234-567812345678"
    read_runner = ClaudeCodeRunner(
        model="qwen3.5:9b",
        timeout_seconds=60,
        max_turns=1,
        context_length=32768,
        max_output_tokens=8192,
        allow_bash=False,
        available_tools=("Read",),
        session_id=session_id,
        persist_session=True,
    )
    edit_runner = ClaudeCodeRunner(
        model="qwen3.5:9b",
        timeout_seconds=60,
        max_turns=1,
        context_length=32768,
        max_output_tokens=8192,
        allow_bash=False,
        available_tools=("Edit",),
        session_id=session_id,
        resume_session=True,
        persist_session=True,
    )

    read_command = read_runner.command("Read once")
    edit_command = edit_runner.command("Edit once")
    assert read_command[read_command.index("--session-id") + 1] == session_id
    assert edit_command[edit_command.index("--resume") + 1] == session_id
    assert "--no-session-persistence" not in read_command
    assert "--no-session-persistence" not in edit_command
    assert edit_command[edit_command.index("--tools") + 1] == "Edit"


def test_runner_rejects_unscoped_session_continuation() -> None:
    """Session resume must supply a valid UUID and explicitly enable persistence."""

    with pytest.raises(ValueError, match="requires session_id"):
        ClaudeCodeRunner(
            model="qwen3.5:9b",
            timeout_seconds=60,
            max_turns=1,
            context_length=32768,
            max_output_tokens=8192,
            resume_session=True,
            persist_session=True,
        )


def test_reading_visible_test_script_is_not_counted_as_execution() -> None:
    """Reading the scheduler script is only exploration and must not increase any test-call metrics."""

    metrics = ClaudeCodeRunner._summarize(
        [
            {
                "event_type": "assistant",
                "details": {
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "input": {"command": "cat scripts/run_visible_tests.py"},
                            }
                        ]
                    }
                },
            }
        ],
        timed_out=False,
    )

    assert metrics["visible_test_executions"] == 0
    assert metrics["agent_test_command_calls"] == 0
    assert metrics["host_test_calls"] == 0
