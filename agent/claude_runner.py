"""Invoke local Ollama through the Claude Code CLI and preserve an auditable observable trajectory.

This module is responsible only for the per-task Agent process boundary. The caller
is responsible for preparing a clean worktree, creating the run directory, and
collecting the final Git patch. The CLI uses structured streaming output; parsing
discards hidden reasoning fields such as thinking and keeps only experiment facts
such as model-visible replies, tool calls, tool results, and usage.
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
from uuid import UUID


class ClaudeCodeError(RuntimeError):
    """Raised when the Claude Code configuration is unsafe, cannot start, or its output is unparseable."""


@dataclass(frozen=True, slots=True)
class ClaudeCodeResult:
    """The complete observable result of one Claude Code process."""

    exit_code: int
    agent_log: str
    test_output: str
    events: tuple[dict[str, Any], ...]
    timed_out: bool
    metrics: dict[str, Any]


def combine_phase_results(
    phases: Sequence[tuple[str, ClaudeCodeResult]],
) -> ClaudeCodeResult:
    """Merge multiple independent Claude sessions while preserving per-phase termination reasons and usage.

    The last verification session decides the overall exit code; the implementation
    phase ending because of the turn budget must not cause a run that was later
    successfully repaired to still be marked as failed. A wall-clock timeout in any
    phase is still recorded separately in metrics, making it easy to analyze whether
    the budget is reasonable.
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
    total_visible_test_parent_generated = 0
    total_visible_test_executions = 0
    total_visible_test_valid_executions = 0
    total_visible_test_infrastructure_errors = 0
    total_visible_test_baseline_failure_signatures = 0
    total_visible_test_candidate_failure_signatures = 0
    total_visible_test_new_failure_signatures = 0
    total_visible_test_outcome_parser_failures = 0
    total_visible_test_passed = 0
    total_visible_test_baseline_executions = 0
    total_visible_test_baseline_passed = 0
    total_visible_test_comparisons = 0
    total_visible_test_new_regressions = 0
    total_visible_test_fixed_baseline_failures = 0
    total_visible_test_unchanged_baseline_failures = 0
    total_visible_test_timed_out = 0
    total_visible_test_duration_seconds = 0.0
    any_cache_hit = False
    any_baseline_timed_out = False
    any_candidate_timed_out = False
    task_budget_exhausted = False
    test_evidence_available = False

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
        total_visible_test_parent_generated += int(
            result.metrics.get("visible_test_parent_generated", 0)
        )
        total_visible_test_executions += int(
            result.metrics.get("visible_test_executions", 0)
        )
        total_visible_test_valid_executions += int(
            result.metrics.get("visible_test_valid_executions", 0)
        )
        total_visible_test_infrastructure_errors += int(
            result.metrics.get("visible_test_infrastructure_errors", 0)
        )
        total_visible_test_baseline_failure_signatures += int(
            result.metrics.get("visible_test_baseline_failure_signatures", 0)
        )
        total_visible_test_candidate_failure_signatures += int(
            result.metrics.get("visible_test_candidate_failure_signatures", 0)
        )
        total_visible_test_new_failure_signatures += int(
            result.metrics.get("visible_test_new_failure_signatures", 0)
        )
        total_visible_test_outcome_parser_failures += int(
            result.metrics.get("visible_test_outcome_parser_failures", 0)
        )
        test_evidence_available = test_evidence_available or bool(
            result.metrics.get("test_evidence_available", False)
        )
        total_visible_test_passed += int(
            result.metrics.get("visible_test_passed", 0)
        )
        total_visible_test_baseline_executions += int(
            result.metrics.get("visible_test_baseline_executions", 0)
        )
        total_visible_test_baseline_passed += int(
            result.metrics.get("visible_test_baseline_passed", 0)
        )
        total_visible_test_comparisons += int(
            result.metrics.get("visible_test_comparisons", 0)
        )
        total_visible_test_new_regressions += int(
            result.metrics.get("visible_test_new_regressions", 0)
        )
        total_visible_test_fixed_baseline_failures += int(
            result.metrics.get("visible_test_fixed_baseline_failures", 0)
        )
        total_visible_test_unchanged_baseline_failures += int(
            result.metrics.get("visible_test_unchanged_baseline_failures", 0)
        )
        total_visible_test_timed_out += int(
            bool(result.metrics.get("visible_test_timed_out", False))
        )
        total_visible_test_duration_seconds += float(
            result.metrics.get("duration_seconds", 0.0)
        )
        any_cache_hit = any_cache_hit or bool(result.metrics.get("cache_hit", False))
        any_baseline_timed_out = any_baseline_timed_out or bool(
            result.metrics.get("baseline_timed_out", False)
        )
        any_candidate_timed_out = any_candidate_timed_out or bool(
            result.metrics.get("candidate_timed_out", False)
        )
        task_budget_exhausted = task_budget_exhausted or bool(
            result.metrics.get("task_budget_exhausted", False)
        )
        usage = result.metrics.get("token_usage", {})
        if isinstance(usage, Mapping):
            for key, value in usage.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    aggregate_usage[str(key)] = aggregate_usage.get(str(key), 0) + value

    metrics: dict[str, Any] = {
        "agent_turns": total_turns,
        "tool_calls": total_tool_calls,
        # The legacy field is kept as a compatibility alias, but it only equals the
        # number of executions for which the scheduler confirmed a container was
        # created; it must no longer be inferred from script names appearing in Bash
        # text.
        "visible_test_calls": total_visible_test_executions,
        "visible_test_requests": total_visible_test_requests,
        "visible_test_missing": total_visible_test_missing,
        "visible_test_rejected": total_visible_test_rejected,
        "visible_test_parent_generated": total_visible_test_parent_generated,
        "visible_test_executions": total_visible_test_executions,
        "visible_test_valid_executions": total_visible_test_valid_executions,
        "visible_test_infrastructure_errors": (
            total_visible_test_infrastructure_errors
        ),
        "visible_test_baseline_failure_signatures": (
            total_visible_test_baseline_failure_signatures
        ),
        "visible_test_candidate_failure_signatures": (
            total_visible_test_candidate_failure_signatures
        ),
        "visible_test_new_failure_signatures": (
            total_visible_test_new_failure_signatures
        ),
        "visible_test_outcome_parser_failures": (
            total_visible_test_outcome_parser_failures
        ),
        "test_evidence_available": test_evidence_available,
        "visible_test_passed": total_visible_test_passed,
        "visible_test_baseline_executions": total_visible_test_baseline_executions,
        "visible_test_baseline_passed": total_visible_test_baseline_passed,
        "visible_test_comparisons": total_visible_test_comparisons,
        "visible_test_new_regressions": total_visible_test_new_regressions,
        "visible_test_fixed_baseline_failures": (
            total_visible_test_fixed_baseline_failures
        ),
        "visible_test_unchanged_baseline_failures": (
            total_visible_test_unchanged_baseline_failures
        ),
        "visible_test_timed_out": total_visible_test_timed_out,
        # duration_seconds only counts the time this round actually waited on Docker;
        # a cache hit returns 0, and the full per-task wall clock still uses
        # metadata.runtime_seconds as the source of truth.
        "duration_seconds": round(total_visible_test_duration_seconds, 6),
        "cache_hit": any_cache_hit,
        "baseline_timed_out": any_baseline_timed_out,
        "candidate_timed_out": any_candidate_timed_out,
        "task_budget_exhausted": task_budget_exhausted,
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
    """Return the current UTC time with a timezone, for uniform use across trajectory events."""

    return datetime.now(timezone.utc).isoformat()


def _sanitize_value(value: Any) -> Any:
    """Recursively remove hidden reasoning fields while keeping the visible data needed for a reproducible experiment."""

    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_value(item)
            for key, item in value.items()
            if str(key).lower() not in {"thinking", "reasoning", "chain_of_thought"}
        }
    if isinstance(value, list):
        sanitized_items = []
        for item in value:
            # Claude stream thinking blocks are identified by their type; deleting the
            # whole block, rather than only clearing the text, better guarantees the
            # trajectory cannot imply the model's private thought process was saved.
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
    """Recognize tests the model actually tried to execute, rather than substring-counting script names.

    ``cat scripts/run_visible_tests.py`` must not be counted as a test; directly
    executing that script, the repository-specific ``runtests.py``/``bin/test``, and
    the common test entry points are what count as model-initiated host test attempts.
    This metric is used only for compliance analysis; the real visible execution comes
    from the Docker scheduling result.
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
    """Run Claude Code with fixed CLI arguments while enforcing local-model and outbound restrictions."""

    def __init__(
        self,
        *,
        model: str,
        timeout_seconds: float,
        max_turns: int,
        context_length: int,
        max_output_tokens: int,
        base_url: str = "http://localhost:11434",
        executable: str = "claude",
        allow_bash: bool = True,
        available_tools: Sequence[str] | None = None,
        session_id: str | None = None,
        resume_session: bool = False,
        persist_session: bool = False,
    ) -> None:
        """Validate experiment limits, tool gating, transient session state, and the local endpoint."""

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
        if resume_session and session_id is None:
            raise ValueError("resume_session requires session_id")
        if (session_id is not None or resume_session) and not persist_session:
            raise ValueError("session continuation requires persist_session")
        if session_id is not None:
            try:
                UUID(session_id)
            except ValueError as error:
                raise ValueError("session_id must be a valid UUID") from error
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
        # The allowlist is passed in only from the parent's fixed policy; preserving
        # first-occurrence order makes the CLI command easier to audit.
        self.available_tools = (
            tuple(dict.fromkeys(available_tools))
            if available_tools is not None
            else None
        )
        self.session_id = session_id
        self.resume_session = resume_session
        self.persist_session = persist_session

    def command(self, prompt: str) -> list[str]:
        """Build a fixed command without shell interpolation, so problem text cannot be interpreted as a command."""

        disallowed_tools = ["WebFetch", "WebSearch"]
        if not self.allow_bash:
            # Verification/Recovery only allow built-in read/write tools; real tests
            # are executed by the parent process inside Docker, and CLI-level disabling
            # blocks host commands more effectively than a Prompt soft constraint.
            disallowed_tools.append("Bash")
        disallowed_tools = list(dict.fromkeys(disallowed_tools))
        command = [
            self.executable,
            "--print",
            # prompt follows the fixed-arity arguments and must not come after the
            # variable-length --disallowed-tools argument, otherwise the CLI parser
            # could mistake the problem text for another tool rule.
            prompt,
            "--bare",
            "--verbose",
            "--output-format",
            "stream-json",
            "--model",
            self.model,
            "--max-turns",
            str(self.max_turns),
        ]
        if self.available_tools is not None:
            # ``--tools`` is the real available-tools allowlist. Pass a single
            # comma-separated argument so its variable-length parsing does not swallow
            # the fixed CLI options that follow.
            command.extend(("--tools", ",".join(self.available_tools)))
        if self.session_id is not None:
            # The Recovery gate's Read and Edit are two CLI invocations with different
            # tool allowlists; an explicit session ID lets the second step inherit the
            # context of files read in the first step, while avoiding ``--continue``
            # accidentally attaching to the user's or another task's most recent session.
            command.extend(
                (
                    "--resume" if self.resume_session else "--session-id",
                    self.session_id,
                )
            )
        session_options = (
            () if self.persist_session else ("--no-session-persistence",)
        )
        command.extend(
            (
                "--permission-mode",
                "bypassPermissions",
                *session_options,
                "--no-chrome",
                "--disable-slash-commands",
                "--disallowed-tools",
                *disallowed_tools,
            )
        )
        return command

    def environment(self, source: Mapping[str, str] | None = None) -> dict[str, str]:
        """Build a subprocess environment that strips cloud credentials and limits common outbound paths."""

        environment = dict(os.environ if source is None else source)
        # Never pass a possibly-present Anthropic cloud key to the Agent; Ollama only
        # requires the token to be non-empty.
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
                # A local model absent from the Claude Code model catalog may be given
                # an output budget close to the whole context by default; only an
                # explicit limit reserves stable space for the Prompt and tool results.
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(self.max_output_tokens),
                "CLAUDE_CODE_MAX_TURNS": str(self.max_turns),
                "MCP_CONNECTION_NONBLOCKING": "true",
                # Claude Code 2.1.280 does not reliably honor HTTP NO_PROXY; to keep
                # local Ollama reachable, only intercept HTTPS, which usually carries
                # GitHub/PyPI access. Web tools and cloud credentials are disabled
                # separately, but this still does not pretend to be kernel-firewall-level
                # isolation.
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "https_proxy": "http://127.0.0.1:9",
                "NO_PROXY": "localhost,127.0.0.1,::1",
                "no_proxy": "localhost,127.0.0.1,::1",
            }
        )
        return environment

    def run(self, repository: Path | str, prompt: str) -> ClaudeCodeResult:
        """Run the Agent in the target worktree and terminate the whole process group on wall-clock timeout."""

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
        """Compute the metrics needed for reporting from the sanitized events, without relying on the model's success claims."""

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
            # Claude Code's final result event provides the authoritative usage; read
            # only that event so the per-message incremental usage is not added twice
            # to the final cumulative value.
            if event_type == "result":
                raw_usage = details.get("usage")
                if isinstance(raw_usage, Mapping):
                    usage = {
                        str(key): value
                        for key, value in raw_usage.items()
                        if isinstance(value, int) and not isinstance(value, bool)
                    }
                # Preserve the observable termination classification Claude Code
                # provides directly, avoiding having to guess root causes such as
                # max_turns or blocking_limit from agent.log text afterwards.
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
            "agent_test_command_calls": agent_test_command_calls,
            "host_test_calls": host_test_calls,
            "timed_out": timed_out,
            "token_usage": usage,
        }
        summary.update(terminal)
        return summary

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
        """On timeout, terminate Claude and its Bash subprocesses so no test container lingers in the background."""

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
        """Convert JSONL into observable events and separately extract the test-related command output."""

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
            # stderr is public process diagnostic information and contains no Claude
            # stream thinking blocks.
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
        """Correlate test Bash calls with their tool_results across message blocks to form a test log."""

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
