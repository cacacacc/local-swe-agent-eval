"""验证 Claude Code 命令边界、外联限制、轨迹清洗和测试输出提取。"""

import json

import pytest

from agent.claude_runner import ClaudeCodeError, ClaudeCodeRunner


def make_runner() -> ClaudeCodeRunner:
    """构造与正式实验关键上限一致、但不会实际启动进程的 runner。"""

    return ClaudeCodeRunner(
        model="qwen3.5:9b",
        timeout_seconds=5400,
        max_turns=30,
        context_length=32768,
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
        "timed_out": False,
        "token_usage": {"input_tokens": 120, "output_tokens": 30},
    }
