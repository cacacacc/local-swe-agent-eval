"""通过 Claude Code CLI 调用本机 Ollama，并保留可审计的可观察轨迹。

本模块只负责单道题的 Agent 进程边界。调用方负责准备干净 worktree、创建运行
目录和收集最终 Git patch。CLI 使用结构化流输出；解析时会丢弃 thinking 等隐藏
推理字段，只保存模型可见回复、工具调用、工具结果和用量等实验事实。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


class ClaudeCodeError(RuntimeError):
    """当 Claude Code 配置不安全、无法启动或输出不可解析时抛出。"""


@dataclass(frozen=True, slots=True)
class ClaudeCodeResult:
    """一次 Claude Code 进程的完整可观察结果。"""

    exit_code: int
    agent_log: str
    test_output: str
    events: tuple[dict[str, Any], ...]
    timed_out: bool
    metrics: dict[str, Any]


def combine_phase_results(
    phases: Sequence[tuple[str, ClaudeCodeResult]],
) -> ClaudeCodeResult:
    """合并多个独立 Claude 会话，同时保留逐阶段终止原因与用量。

    最后一个验证会话决定总体 exit code；实现阶段因 turn 预算结束并不应让一个
    后续已成功修复的运行仍被标记为失败。任何阶段的墙钟超时仍单独记录在 metrics，
    便于分析预算是否合理。
    """

    if not phases:
        raise ValueError("at least one Claude Code phase is required")

    events: list[dict[str, Any]] = []
    agent_logs: list[str] = []
    test_outputs: list[str] = []
    phase_metrics: dict[str, dict[str, Any]] = {}
    aggregate_usage: dict[str, int] = {}
    total_turns = 0
    total_tool_calls = 0
    total_host_test_calls = 0
    total_agent_test_commands = 0
    total_visible_test_requests = 0
    total_visible_test_missing = 0
    total_visible_test_rejected = 0
    total_visible_test_executions = 0
    total_visible_test_passed = 0
    total_visible_test_timed_out = 0

    for phase_name, result in phases:
        events.append(
            {
                "sequence": len(events) + 1,
                "timestamp": _utc_now(),
                "event_type": "phase_start",
                "phase": phase_name,
                "details": {"phase": phase_name},
            }
        )
        for event in result.events:
            copied = dict(event)
            copied["sequence"] = len(events) + 1
            copied["phase"] = phase_name
            events.append(copied)

        agent_logs.append(f"===== phase: {phase_name} =====\n{result.agent_log}")
        if result.test_output:
            test_outputs.append(
                f"===== phase: {phase_name} =====\n{result.test_output}"
            )
        phase_metrics[phase_name] = dict(result.metrics)
        total_turns += int(result.metrics.get("agent_turns", 0))
        total_tool_calls += int(result.metrics.get("tool_calls", 0))
        total_host_test_calls += int(result.metrics.get("host_test_calls", 0))
        total_agent_test_commands += int(
            result.metrics.get("agent_test_command_calls", 0)
        )
        total_visible_test_requests += int(
            result.metrics.get("visible_test_requests", 0)
        )
        total_visible_test_missing += int(
            result.metrics.get("visible_test_missing", 0)
        )
        total_visible_test_rejected += int(
            result.metrics.get("visible_test_rejected", 0)
        )
        total_visible_test_executions += int(
            result.metrics.get("visible_test_executions", 0)
        )
        total_visible_test_passed += int(
            result.metrics.get("visible_test_passed", 0)
        )
        total_visible_test_timed_out += int(
            bool(result.metrics.get("visible_test_timed_out", False))
        )
        usage = result.metrics.get("token_usage", {})
        if isinstance(usage, Mapping):
            for key, value in usage.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    aggregate_usage[str(key)] = aggregate_usage.get(str(key), 0) + value

    metrics: dict[str, Any] = {
        "agent_turns": total_turns,
        "tool_calls": total_tool_calls,
        # 旧字段保留为兼容别名，但只等于调度器确认创建过容器的执行次数，不能再
        # 通过 Bash 文本中出现脚本名称来推测。
        "visible_test_calls": total_visible_test_executions,
        "visible_test_requests": total_visible_test_requests,
        "visible_test_missing": total_visible_test_missing,
        "visible_test_rejected": total_visible_test_rejected,
        "visible_test_executions": total_visible_test_executions,
        "visible_test_passed": total_visible_test_passed,
        "visible_test_timed_out": total_visible_test_timed_out,
        "agent_test_command_calls": total_agent_test_commands,
        "host_test_calls": total_host_test_calls,
        "timed_out": any(result.timed_out for _, result in phases),
        "token_usage": aggregate_usage,
        "phases": phase_metrics,
    }
    return ClaudeCodeResult(
        exit_code=phases[-1][1].exit_code,
        agent_log="\n".join(agent_logs),
        test_output="\n".join(test_outputs),
        events=tuple(events),
        timed_out=bool(metrics["timed_out"]),
        metrics=metrics,
    )


def _utc_now() -> str:
    """返回带时区的 UTC 时间，供轨迹事件统一使用。"""

    return datetime.now(timezone.utc).isoformat()


def _sanitize_value(value: Any) -> Any:
    """递归移除隐藏推理字段，同时保留可复现实验所需的可见数据。"""

    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_value(item)
            for key, item in value.items()
            if str(key).lower() not in {"thinking", "reasoning", "chain_of_thought"}
        }
    if isinstance(value, list):
        sanitized_items = []
        for item in value:
            # Claude stream 的 thinking block 以 type 标识；整块删除比只清空文本
            # 更能保证 trajectory 不会暗示保存了模型私有思维过程。
            if isinstance(item, Mapping) and item.get("type") in {
                "thinking",
                "redacted_thinking",
            }:
                continue
            sanitized_items.append(_sanitize_value(item))
        return sanitized_items
    return value


_TEST_COMMAND_PATTERN = re.compile(
    r"(?:^|[;&|]\s*)"
    r"(?:\S*python\S*\s+-m\s+)?"
    r"(?:pytest|tox|sphinx-build|npm\s+test)(?:\s|$)",
    flags=re.IGNORECASE,
)


def _looks_like_test_command(command: str) -> bool:
    """识别模型实际尝试执行的测试，而不是对脚本名称做子串计数。

    ``cat scripts/run_visible_tests.py`` 不应被记成测试；直接执行该脚本、仓库专用
    ``runtests.py``/``bin/test`` 以及常见测试入口才属于模型发起的宿主测试尝试。
    该指标只用于合规分析，真正的 visible execution 由 Docker 调度结果产生。
    """

    stripped = command.strip()
    if _TEST_COMMAND_PATTERN.search(stripped):
        return True
    first_segment = re.split(r"[;&|]", stripped, maxsplit=1)[0].strip()
    tokens = first_segment.split()
    if not tokens:
        return False
    executable = Path(tokens[0]).name.lower()
    if executable in {"cat", "head", "less", "more", "rg", "sed", "tail"}:
        return False
    return any(
        token.endswith("run_visible_tests.py")
        or token.endswith("runtests.py")
        or token.rstrip("/").endswith("bin/test")
        for token in tokens
    )


class ClaudeCodeRunner:
    """以固定 CLI 参数执行 Claude Code，并实施本地模型与外联限制。"""

    def __init__(
        self,
        *,
        model: str,
        timeout_seconds: int,
        max_turns: int,
        context_length: int,
        max_output_tokens: int,
        base_url: str = "http://localhost:11434",
        executable: str = "claude",
        allow_bash: bool = True,
    ) -> None:
        """验证实验上限和 Ollama endpoint，拒绝把请求发往远程主机。"""

        if (
            timeout_seconds <= 0
            or max_turns <= 0
            or context_length <= 0
            or max_output_tokens <= 0
        ):
            raise ValueError(
                "timeout, max_turns, context_length and max_output_tokens must be positive"
            )
        if max_output_tokens >= context_length:
            raise ValueError("max_output_tokens must be smaller than context_length")
        parsed_url = urlparse(base_url)
        if parsed_url.scheme != "http" or parsed_url.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ClaudeCodeError(
                "ANTHROPIC_BASE_URL must be a local HTTP Ollama endpoint"
            )

        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_turns = max_turns
        self.context_length = context_length
        self.max_output_tokens = max_output_tokens
        self.base_url = base_url.rstrip("/")
        self.executable = executable
        self.allow_bash = allow_bash

    def command(self, prompt: str) -> list[str]:
        """构造无 shell 插值的固定命令，避免题目文本被解释为命令。"""

        disallowed_tools = ["WebFetch", "WebSearch"]
        if not self.allow_bash:
            # 验证会话只能 Read/Edit；真正的测试由父进程在 Docker 中执行，CLI
            # 级禁用比 Prompt 软约束更能防止小模型回到宿主 pytest/pip。
            disallowed_tools.append("Bash")
        return [
            self.executable,
            "--print",
            # prompt 紧跟固定元数参数，不能放在可变长的 --disallowed-tools
            # 参数之后，否则 CLI 解析器可能把题目文本误当成另一个工具规则。
            prompt,
            "--bare",
            "--verbose",
            "--output-format",
            "stream-json",
            "--model",
            self.model,
            "--max-turns",
            str(self.max_turns),
            "--permission-mode",
            "bypassPermissions",
            "--no-session-persistence",
            "--no-chrome",
            "--disable-slash-commands",
            "--disallowed-tools",
            *disallowed_tools,
        ]

    def environment(self, source: Mapping[str, str] | None = None) -> dict[str, str]:
        """生成最小化云凭据且限制常见外联路径的子进程环境。"""

        environment = dict(os.environ if source is None else source)
        # 绝不把可能存在的 Anthropic 云密钥传给 Agent；Ollama 只要求 token 非空。
        for name in (
            "ANTHROPIC_API_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "HTTP_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "all_proxy",
        ):
            environment.pop(name, None)
        environment.update(
            {
                "ANTHROPIC_AUTH_TOKEN": "ollama",
                "ANTHROPIC_BASE_URL": self.base_url,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_LOGIN_COMMAND": "1",
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(self.context_length),
                # 未收录于 Claude Code model catalog 的本地模型可能被默认赋予接近整个
                # context 的输出预算；显式限制后才能给 Prompt 和工具结果保留稳定空间。
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(self.max_output_tokens),
                "CLAUDE_CODE_MAX_TURNS": str(self.max_turns),
                "MCP_CONNECTION_NONBLOCKING": "true",
                # Claude Code 2.1.280 不可靠地遵循 HTTP NO_PROXY；为保证本机
                # Ollama 可达，只拦截通常承载 GitHub/PyPI 等访问的 HTTPS。
                # Web 工具和云凭据另行禁用，但这仍不冒充内核防火墙级隔离。
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "https_proxy": "http://127.0.0.1:9",
                "NO_PROXY": "localhost,127.0.0.1,::1",
                "no_proxy": "localhost,127.0.0.1,::1",
            }
        )
        return environment

    def run(self, repository: Path | str, prompt: str) -> ClaudeCodeResult:
        """在目标 worktree 运行 Agent，并在墙钟超时后终止整个进程组。"""

        repository_path = Path(repository).resolve()
        if not repository_path.is_dir():
            raise ClaudeCodeError(f"repository does not exist: {repository_path}")

        try:
            process = subprocess.Popen(
                self.command(prompt),
                cwd=repository_path,
                env=self.environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=(os.name != "nt"),
            )
        except OSError as error:
            raise ClaudeCodeError(f"cannot start Claude Code: {error}") from error

        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._terminate_process_tree(process)
            stdout, stderr = process.communicate()

        events, normalized_log, test_output = self._parse_stream(stdout, stderr)
        if timed_out:
            events.append(
                {
                    "sequence": len(events) + 1,
                    "timestamp": _utc_now(),
                    "event_type": "agent_timeout",
                    "details": {"timeout_seconds": self.timeout_seconds},
                }
            )
        return ClaudeCodeResult(
            exit_code=124 if timed_out else process.returncode,
            agent_log=normalized_log,
            test_output=test_output,
            events=tuple(events),
            timed_out=timed_out,
            metrics=self._summarize(events, timed_out),
        )

    @staticmethod
    def _summarize(
        events: Sequence[Mapping[str, Any]],
        timed_out: bool,
    ) -> dict[str, Any]:
        """从清洗后的事件计算报告所需指标，不依赖模型的成功声明。"""

        turns = 0
        tool_calls = 0
        host_test_calls = 0
        agent_test_command_calls = 0
        usage: dict[str, int] = {}
        terminal: dict[str, Any] = {}
        for event in events:
            event_type = event.get("event_type")
            details = event.get("details")
            if event_type == "assistant":
                turns += 1
            if not isinstance(details, Mapping):
                continue
            message = details.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
                    for block in content:
                        if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                            continue
                        tool_calls += 1
                        if block.get("name") != "Bash":
                            continue
                        tool_input = block.get("input")
                        command = (
                            tool_input.get("command")
                            if isinstance(tool_input, Mapping)
                            else None
                        )
                        if not isinstance(command, str):
                            continue
                        if _looks_like_test_command(command):
                            agent_test_command_calls += 1
                            host_test_calls += 1
            # Claude Code 的最终 result 事件提供权威用量；只读取该事件，避免把
            # 每条 message 的增量 usage 与最终累计值重复相加。
            if event_type == "result":
                raw_usage = details.get("usage")
                if isinstance(raw_usage, Mapping):
                    usage = {
                        str(key): value
                        for key, value in raw_usage.items()
                        if isinstance(value, int) and not isinstance(value, bool)
                    }
                # 直接保存 Claude Code 给出的可观察终止分类，避免事后只能从
                # agent.log 文本猜测 max_turns、blocking_limit 等根因。
                for source_key, destination_key in (
                    ("terminal_reason", "terminal_reason"),
                    ("subtype", "result_subtype"),
                    ("num_turns", "model_turns"),
                ):
                    value = details.get(source_key)
                    if isinstance(value, (str, int)) and not isinstance(value, bool):
                        terminal[destination_key] = value
        summary = {
            "agent_turns": turns,
            "tool_calls": tool_calls,
            "visible_test_calls": 0,
            "visible_test_requests": 0,
            "visible_test_missing": 0,
            "visible_test_rejected": 0,
            "visible_test_executions": 0,
            "visible_test_passed": 0,
            "visible_test_timed_out": False,
            "agent_test_command_calls": agent_test_command_calls,
            "host_test_calls": host_test_calls,
            "timed_out": timed_out,
            "token_usage": usage,
        }
        summary.update(terminal)
        return summary

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
        """超时时终止 Claude 及其 Bash 子进程，避免测试容器在后台残留。"""

        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
                return
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    return
                except OSError:
                    pass
        process.kill()

    @staticmethod
    def _parse_stream(
        stdout: str,
        stderr: str,
    ) -> tuple[list[dict[str, Any]], str, str]:
        """把 JSONL 转成可观察事件，并单独提取测试相关命令输出。"""

        events: list[dict[str, Any]] = []
        normalized_lines: list[str] = []
        test_lines: list[str] = []
        test_tool_ids: set[str] = set()
        sequence = 0

        for raw_line in stdout.splitlines():
            if not raw_line.strip():
                continue
            sequence += 1
            try:
                raw_event = json.loads(raw_line)
            except json.JSONDecodeError:
                sanitized: Any = {"text": raw_line, "malformed_json": True}
                event_type = "unstructured_output"
            else:
                sanitized = _sanitize_value(raw_event)
                event_type = str(raw_event.get("type", "unknown"))
                ClaudeCodeRunner._track_test_tools(
                    sanitized,
                    test_tool_ids,
                    test_lines,
                )
            normalized_lines.append(
                json.dumps(sanitized, ensure_ascii=False, sort_keys=True)
            )
            events.append(
                {
                    "sequence": sequence,
                    "timestamp": _utc_now(),
                    "event_type": event_type,
                    "details": sanitized,
                }
            )

        if stderr.strip():
            sequence += 1
            # stderr 是公开的进程诊断信息，不包含 Claude stream 的 thinking block。
            events.append(
                {
                    "sequence": sequence,
                    "timestamp": _utc_now(),
                    "event_type": "process_stderr",
                    "details": {"text": stderr},
                }
            )
            normalized_lines.append(
                json.dumps(
                    {"type": "process_stderr", "text": stderr},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

        agent_log = "\n".join(normalized_lines)
        if agent_log:
            agent_log += "\n"
        test_output = "\n".join(test_lines)
        if test_output:
            test_output += "\n"
        return events, agent_log, test_output

    @staticmethod
    def _track_test_tools(
        event: Any,
        test_tool_ids: set[str],
        test_lines: list[str],
    ) -> None:
        """从消息块中关联测试 Bash 调用及其 tool_result，形成测试日志。"""

        if not isinstance(event, Mapping):
            return
        message = event.get("message")
        if not isinstance(message, Mapping):
            return
        content = message.get("content")
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            return

        for block in content:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") == "tool_use" and block.get("name") == "Bash":
                tool_input = block.get("input")
                command = tool_input.get("command") if isinstance(tool_input, Mapping) else None
                if isinstance(command, str) and _looks_like_test_command(command):
                    tool_id = block.get("id")
                    if isinstance(tool_id, str):
                        test_tool_ids.add(tool_id)
                    test_lines.append(f"$ {command}")
            elif block.get("type") == "tool_result":
                tool_id = block.get("tool_use_id")
                if tool_id in test_tool_ids:
                    test_lines.append(str(block.get("content", "")))
