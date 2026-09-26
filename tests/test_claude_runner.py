"""验证 Claude Code 命令边界、外联限制、轨迹清洗和测试输出提取。"""

import json

import pytest

from agent.claude_runner import (
    ClaudeCodeError,
    ClaudeCodeResult,
    ClaudeCodeRunner,
    combine_phase_results,
)


def make_runner() -> ClaudeCodeRunner:
    """构造与正式实验关键上限一致、但不会实际启动进程的 runner。"""

    return ClaudeCodeRunner(
        model="qwen3.5:9b",
        timeout_seconds=5400,
        max_turns=30,
        context_length=32768,
        max_output_tokens=8192,
    )


def test_command_fixes_model_turn_limit_and_noninteractive_isolation() -> None:
    """命令必须固定本地模型、turn 上限，并隔离用户插件与会话历史。"""

    command = make_runner().command("Solve the task")

    assert command[0] == "claude"
    assert command[command.index("--print") + 1] == "Solve the task"
    assert command[command.index("--model") + 1] == "qwen3.5:9b"
    assert command[command.index("--max-turns") + 1] == "30"
    assert "--bare" in command
    assert "--no-session-persistence" in command
    assert "WebFetch" in command and "WebSearch" in command
    # Prompt 必须位于可变长的工具列表之前，防止被 CLI 当成规则吞掉。
    assert command.index("Solve the task") < command.index("--disallowed-tools")


def test_environment_removes_cloud_credentials_and_keeps_only_local_endpoint() -> None:
    """子进程不能继承云密钥，常见网络客户端也只能绕过代理访问本机。"""

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
    """配置错误时必须在启动 Agent 前拒绝远程模型服务。"""

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
    """轨迹不得保存隐藏推理，但必须关联测试命令和对应工具结果。"""

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
    """报告指标必须来自结构化事件，并采用最终 result 的累计 token usage。"""

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
        "visible_test_rejected": 0,
        "visible_test_executions": 0,
        "visible_test_passed": 0,
        "visible_test_timed_out": False,
        "agent_test_command_calls": 0,
        "host_test_calls": 0,
        "timed_out": False,
        "token_usage": {"input_tokens": 120, "output_tokens": 30},
    }


def test_summary_preserves_observable_terminal_reason() -> None:
    """终止原因必须进入结构化指标，避免分析时依赖截断后的日志文本。"""

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
    """实现阶段耗尽后，独立验证阶段仍可修复，并保留两个阶段的审计指标。"""

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


def test_summary_never_infers_visible_execution_from_bash_text() -> None:
    """模型 Bash 只能记为宿主尝试，脚本名称不能冒充 Docker 执行。"""

    events = []
    for command in (
        "python scripts/run_visible_tests.py -- python -m pytest tests/test_one.py",
        "/host/venv/bin/python -m pytest tests/test_two.py",
    ):
        events.append(
            {
                "event_type": "assistant",
                "details": {
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "input": {"command": command},
                            }
                        ]
                    }
                },
            }
        )

    metrics = ClaudeCodeRunner._summarize(events, timed_out=False)

    assert metrics["visible_test_calls"] == 0
    assert metrics["visible_test_executions"] == 0
    assert metrics["agent_test_command_calls"] == 2
    assert metrics["host_test_calls"] == 2


def test_verification_runner_disables_bash_at_cli_boundary() -> None:
    """验证会话必须由 CLI 禁用 Bash，不能只依赖模型遵循 Prompt。"""

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


def test_reading_visible_test_script_is_not_counted_as_execution() -> None:
    """读取调度脚本只是探索行为，不能增加任何测试调用指标。"""

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
