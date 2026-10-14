"""Use a local Ollama to drive Claude Code and solve a prepared SWE-bench task.

The input task file must be JSON/JSONL safely projected through ``SWEbenchLoader``; raw data
records containing the gold patch or test patch must not be passed directly to this script.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import time
from typing import Any, Mapping, MutableMapping, Sequence

from agent.claude_runner import (
    ClaudeCodeResult,
    ClaudeCodeRunner,
    combine_phase_results,
)
from agent.handoff import HandoffLedger
from agent.issue_reproduction import (
    IssueReproductionRequest,
    consume_issue_reproduction,
)
from agent.prompt_builder import (
    PromptBuilder,
    build_implementation_phase_prompt,
    build_recovery_read_edit_gate_prompt,
    build_recovery_implementation_prompt,
    build_verification_phase_prompt,
)
from agent.test_plan import (
    TestPlanRequest,
    consume_test_plan,
    generate_repository_test_plan,
)
from agent.test_outcome import TestOutcome, parse_test_outcome
from agent.test_sandbox import (
    TestSandboxError,
    VisibleTestResult,
    VisibleTestSandbox,
    truncate_output,
)
from benchmark.repo_manager import RepositoryManager
from benchmark.swebench_loader import SWEbenchLoader
from benchmark.task import SWEbenchTask
from experiment.config import ExperimentConfig
from experiment.runtime_fingerprint import RuntimeFingerprintCollector
from tracking.run_manager import RunManager, RunSession


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".jinja",
    ".jinja2",
    ".js",
    ".jsx",
    ".php",
    ".pxd",
    ".py",
    ".pyi",
    ".pyx",
    ".rb",
    ".rs",
    ".scss",
    ".ts",
    ".tsx",
}


class TaskBudgetExhausted(RuntimeError):
    """The per-task total wall-clock budget is exhausted; the caller must still save the patch and trajectory already produced."""


@dataclass(frozen=True, slots=True)
class _TaskBudget:
    """Use a monotonic clock to constrain the scattered model and Docker timeouts under a single deadline."""

    deadline_monotonic: float | None

    @classmethod
    def start(cls, timeout_seconds: int) -> "_TaskBudget":
        """Start the budget from the current moment; 0 is the compatibility off value used by older configs."""

        deadline = (
            time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
        )
        return cls(deadline_monotonic=deadline)

    def limit(self, requested_seconds: float) -> float:
        """Return a sub-phase budget that never crosses the total deadline, aborting scheduling immediately when exhausted."""

        if requested_seconds <= 0:
            raise ValueError("requested timeout must be positive")
        if self.deadline_monotonic is None:
            return requested_seconds
        remaining = self.deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise TaskBudgetExhausted("task wall-clock budget exhausted")
        # subprocess accepts fractional seconds; keeping a millisecond margin avoids artificially rounding up by one second when less than a second remains.
        return min(requested_seconds, max(0.001, remaining))

    def ensure_remaining(self) -> None:
        """After a phase ends, prevent the next model or Docker process from crossing the per-task deadline."""

        if (
            self.deadline_monotonic is not None
            and time.monotonic() >= self.deadline_monotonic
        ):
            raise TaskBudgetExhausted("task wall-clock budget exhausted")


_TestCacheKey = tuple[str, str, tuple[str, ...], str, float]
_TestResultCache = MutableMapping[_TestCacheKey, VisibleTestResult]


def _runner(
    config: ExperimentConfig,
    *,
    turns: int,
    timeout_seconds: float,
    base_url: str,
    allow_bash: bool = True,
    available_tools: Sequence[str] | None = None,
    session_id: str | None = None,
    resume_session: bool = False,
    persist_session: bool = False,
) -> ClaudeCodeRunner:
    """Construct Claude Code according to the phase budget and pass tool and temporary-session boundaries."""

    return ClaudeCodeRunner(
        model=config.model.name,
        timeout_seconds=timeout_seconds,
        max_turns=turns,
        context_length=config.model.context_length,
        max_output_tokens=config.model.max_output_tokens,
        base_url=base_url,
        allow_bash=allow_bash,
        available_tools=available_tools,
        session_id=session_id,
        resume_session=resume_session,
        persist_session=persist_session,
    )


def _apply_patch_gate(result: ClaudeCodeResult, patch: str) -> ClaudeCodeResult:
    """Change an empty patch from "normal completion" into an explicit failure, and record whether test evidence exists.

    This gate does not treat "tests were run" as success, because some repositories lack dependencies in the host environment; the true
    resolved status is still decided solely by the official harness. It only eliminates the false completion caused by the model's generic replies.
    """

    patch_generated = bool(patch.strip())
    existing_source_modified = _patch_modifies_existing_source(patch)
    metrics = dict(result.metrics)
    visible_executions = int(metrics.get("visible_test_executions", 0))
    host_attempts = int(metrics.get("host_test_calls", 0))
    metrics["patch_gate"] = {
        "patch_generated": patch_generated,
        "existing_source_modified": existing_source_modified,
        # Host tests do not use the SWE-bench instance environment and can never satisfy the test gate.
        "test_attempted": visible_executions > 0,
        "visible_test_attempted": visible_executions > 0,
        "host_test_attempted": host_attempts > 0,
        "host_test_valid": False,
    }
    events = list(result.events)
    events.append(
        {
            "sequence": len(events) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "patch_validation",
            "details": metrics["patch_gate"],
        }
    )
    exit_code = result.exit_code
    if (not patch_generated or not existing_source_modified) and exit_code == 0:
        # 2 means the runner's deliverable gate failed, distinct from the Claude CLI's own exit code 1.
        exit_code = 2
    return replace(result, exit_code=exit_code, events=tuple(events), metrics=metrics)


def _patch_modifies_existing_source(patch: str) -> bool:
    """Determine whether the patch modifies at least one existing non-test source file.

    This gate specifically intercepts exploration results that leave only reproduction scripts, temporary text, or newly added test directories. Modifying existing test
    files also does not count as a product fix, preventing the model from bypassing delivery requirements by editing tests.
    """

    return bool(_existing_source_patch_projection(patch))


def _select_source_checkpoint(
    final_patch: str,
    checkpoint_patch: str,
    result: ClaudeCodeResult,
    handoff: HandoffLedger,
    session: RunSession,
    *,
    restore_broken_syntax: bool = False,
) -> tuple[str, ClaudeCodeResult]:
    """Deliver a saved source candidate if later work erased it or introduced syntax errors."""

    if not checkpoint_patch or (
        _patch_modifies_existing_source(final_patch) and not restore_broken_syntax
    ):
        return final_patch, result
    # The saved patch was captured before Verification, outside the worktree. Selecting
    # it does not rewrite the model's files or conceal the final worktree trajectory.
    reason = (
        "Verification introduced a Docker-confirmed Python syntax error"
        if restore_broken_syntax
        else "Verification left no existing product-source modification"
    )
    handoff.record_selection("candidate_restored", patch=checkpoint_patch, reason=reason)
    session.write_handoff(handoff.snapshot())
    metrics = dict(result.metrics)
    metrics["candidate_restored"] = True
    events = tuple(result.events) + ({
        "sequence": len(result.events) + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "candidate_restored",
        "details": {"reason": reason},
    },)
    return checkpoint_patch, replace(result, metrics=metrics, events=events)


def _existing_source_patch_projection(patch: str) -> str:
    """Keep only diffs to existing product source code, for comparing real implementation changes before and after the gate."""

    accepted: list[str] = []
    for section in re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE):
        match = re.match(r"diff --git a/(.+?) b/(.+?)\n", section)
        if match is None or "new file mode " in section:
            continue
        path = Path(match.group(2))
        lowered_parts = {part.lower() for part in path.parts}
        name = path.name.lower()
        is_test = bool(lowered_parts & {"test", "tests", "testing"}) or (
            name.startswith("test_")
            or name.endswith("_test.py")
            or name.endswith(".test.js")
            or name.endswith(".test.ts")
        )
        # Documents, logs, and any existing temporary files likewise cannot satisfy the "fix product source" requirement;
        # the explicit suffix whitelist covers the Python/C/frontend and template sources common in SWE-bench.
        if not is_test and path.suffix.lower() in _SOURCE_SUFFIXES and "@@" in section:
            accepted.append(section)
    return "".join(accepted)


@dataclass(frozen=True, slots=True)
class SyntaxPreflightEvidence:
    """Report whether Docker proved a newly introduced Python syntax error."""

    status: str
    paths: tuple[str, ...] = ()
    output: str = ""
    duration_seconds: float = 0.0

    @property
    def new_syntax_error(self) -> bool:
        """Distinguish an introduced SyntaxError from baseline or runner failures."""

        return self.status == "new_syntax_error"

    def prompt_text(self) -> str:
        """Give the model only bounded, parent-verified syntax evidence."""

        return (
            f"Docker syntax preflight: {self.status}; files={', '.join(self.paths)}\n"
            f"{self.output}"
        )


@dataclass(frozen=True, slots=True)
class IssueReproductionEvidence:
    """Keep baseline/candidate results for a bounded issue-derived script."""

    status: str
    baseline: VisibleTestResult | None = None
    candidate: VisibleTestResult | None = None
    error: str | None = None
    code: str = ""

    def prompt_text(self) -> str:
        """Give Verification concise evidence without pretending an assertion proves full correctness."""

        lines = [f"Issue boundary reproduction: {self.status}"]
        if self.code:
            lines.append(
                "Model-authored assertion (not independent proof):\n"
                f"{truncate_output(self.code, 2200)}"
            )
        if self.error:
            lines.append(f"Runner error: {self.error}")
        for label, result in (("Baseline", self.baseline), ("Candidate", self.candidate)):
            if result is not None:
                lines.append(
                    f"{label}: exit={result.exit_code}; started={result.command_started}; "
                    f"timed_out={result.timed_out}\n{truncate_output(result.output, 900)}"
                )
        return "\n".join(lines)


def _execute_issue_reproduction(
    repository: Path,
    *,
    task: SWEbenchTask,
    base_commit: str,
    request: IssueReproductionRequest,
    sandbox: VisibleTestSandbox,
    task_budget: _TaskBudget,
) -> IssueReproductionEvidence:
    """Compare the model's short assertion on baseline and candidate in offline Docker."""

    if not request.accepted:
        return IssueReproductionEvidence(request.status, error=request.error)
    results: list[VisibleTestResult] = []
    try:
        for apply_patch in (False, True):
            results.append(sandbox.run(
                repository,
                instance_id=task.instance_id,
                base_commit=base_commit,
                command=("python", "-c", request.code),
                apply_patch=apply_patch,
                timeout_seconds=task_budget.limit(min(45, sandbox.timeout_seconds)),
            ))
    except (TestSandboxError, ValueError) as error:
        return IssueReproductionEvidence(
            "unavailable",
            baseline=results[0] if results else None,
            error=str(error)[:800],
            code=request.code,
        )
    baseline, candidate = results
    issue_exceptions = set(re.findall(
        r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b",
        task.problem_statement,
    ))
    baseline_issue_failure = "AssertionError" in baseline.output or any(
        error_type in baseline.output for error_type in issue_exceptions
    )
    if not all(result.evidence_valid and not result.timed_out for result in results):
        status = "unavailable"
    elif baseline.exit_code != 0 and not baseline_issue_failure:
        # An import/setup crash is not evidence that the issue was reproduced. A
        # documented exception (e.g. TypeError from pickle.dumps) is allowed even
        # when it prevents the later assertion from being reached.
        status = "baseline_error"
    elif baseline.exit_code != 0 and candidate.exit_code == 0:
        status = "reproduced_fix"
    elif baseline.exit_code == 0 and candidate.exit_code != 0:
        status = "new_regression"
    elif baseline.exit_code == 0:
        status = "not_reproduced"
    else:
        status = "still_failing"
    return IssueReproductionEvidence(status, baseline, candidate, code=request.code)


def _syntax_preflight(
    repository: Path,
    *,
    task: SWEbenchTask,
    base_commit: str,
    patch: str,
    sandbox: VisibleTestSandbox,
    task_budget: _TaskBudget,
) -> SyntaxPreflightEvidence:
    """Compile existing changed Python source in Docker and compare errors with baseline."""

    paths: list[str] = []
    for section in re.split(
        r"(?=^diff --git )", _existing_source_patch_projection(patch), flags=re.MULTILINE
    ):
        match = re.match(r"diff --git a/(.+?) b/(.+?)\n", section)
        if match is None:
            continue
        name = match.group(2)
        path = Path(name)
        if (
            path.suffix != ".py"
            or path.is_absolute()
            or ".." in path.parts
            or not (repository / path).is_file()
        ):
            continue
        if name not in paths:
            paths.append(name)
    if not paths:
        return SyntaxPreflightEvidence("not_applicable")
    # A large generated patch should not turn a cheap preflight into an unbounded
    # Docker invocation. The regular test phase still covers files beyond this cap.
    paths = paths[:32]
    checker = getattr(sandbox, "syntax_preflight", None)
    if checker is None:
        return SyntaxPreflightEvidence("unavailable", tuple(paths))
    started = time.monotonic()
    try:
        candidate = checker(
            repository,
            instance_id=task.instance_id,
            base_commit=base_commit,
            paths=tuple(paths),
            apply_patch=True,
            timeout_seconds=task_budget.limit(min(60, sandbox.timeout_seconds)),
        )
        if not candidate.evidence_valid or candidate.timed_out:
            status = "unavailable"
        elif candidate.exit_code == 0:
            status = "passed"
        elif re.search(r"(?:SyntaxError|IndentationError|TabError):", candidate.output):
            baseline = checker(
                repository,
                instance_id=task.instance_id,
                base_commit=base_commit,
                paths=tuple(paths),
                apply_patch=False,
                timeout_seconds=task_budget.limit(min(60, sandbox.timeout_seconds)),
            )
            status = (
                "new_syntax_error"
                if baseline.evidence_valid and not baseline.timed_out and baseline.exit_code == 0
                else "unavailable"
            )
        else:
            status = "unavailable"
        output = truncate_output(candidate.output, 2400)
    except (TestSandboxError, ValueError) as error:
        status = "unavailable"
        output = str(error)[:2400]
    return SyntaxPreflightEvidence(
        status, tuple(paths), output, time.monotonic() - started
    )


@dataclass(frozen=True, slots=True)
class _RecoveryGateValidation:
    """The parent process's audit conclusion for a multi-file Recovery gate."""

    valid: bool
    tool_sequence: tuple[str, ...]
    target_files: tuple[str, ...] = ()
    error: str | None = None


def _validate_recovery_read_edit_gate(
    result: ClaudeCodeResult,
    repository: Path,
) -> _RecoveryGateValidation:
    """Accept ordered Reads and Edits of existing product source within the repository.

    Each Edit must target a file read earlier in this gate. This both permits a genuine
    multi-file repair and prevents the relaxed turn budget from becoming an unrestricted
    write session.
    """

    calls = _observable_tool_calls(result)
    sequence = tuple(name for name, _ in calls)
    read_files: set[str] = set()
    edited_files: list[str] = []
    if not calls:
        return _RecoveryGateValidation(
            valid=False,
            tool_sequence=sequence,
            error="gate must contain Read and Edit tool calls",
        )
    for name, tool_input in calls:
        if name not in {"Read", "Edit"}:
            return _RecoveryGateValidation(
                valid=False,
                tool_sequence=sequence,
                target_files=tuple(dict.fromkeys(edited_files)),
                error=f"unsupported gate tool: {name}",
            )
        relative = _validated_gate_source_path(repository, tool_input.get("file_path"))
        if relative is None:
            return _RecoveryGateValidation(
                valid=False,
                tool_sequence=sequence,
                target_files=tuple(dict.fromkeys(edited_files)),
                error=f"{name} target is not an existing product-source file",
            )
        if name == "Read":
            read_files.add(relative)
            continue
        if relative not in read_files:
            return _RecoveryGateValidation(
                valid=False,
                tool_sequence=sequence,
                target_files=tuple(dict.fromkeys((*edited_files, relative))),
                error="Edit target was not read earlier in the gate",
            )
        edited_files.append(relative)
    if not edited_files:
        return _RecoveryGateValidation(
            valid=False,
            tool_sequence=sequence,
            error="gate must contain at least one Edit tool call",
        )
    return _RecoveryGateValidation(
        valid=True,
        tool_sequence=sequence,
        target_files=tuple(dict.fromkeys(edited_files)),
    )


def _observable_tool_calls(
    result: ClaudeCodeResult,
) -> list[tuple[str, Mapping[str, Any]]]:
    """Extract the tools and arguments the model requested in trajectory order, without relying on natural-language claims."""

    calls: list[tuple[str, Mapping[str, Any]]] = []
    for event in result.events:
        if event.get("event_type") != "assistant":
            continue
        details = event.get("details")
        message = details.get("message") if isinstance(details, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue
        for block in content:
            if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                continue
            name = block.get("name")
            tool_input = block.get("input")
            if isinstance(name, str) and isinstance(tool_input, Mapping):
                calls.append((name, tool_input))
    return calls


def _validated_gate_source_path(
    repository: Path,
    raw_path: Any,
) -> str | None:
    """Constrain the tool path to an existing non-test source file inside the worktree and return the POSIX relative path."""

    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    root = repository.resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return None
    lowered_parts = {part.lower() for part in relative.parts}
    name = relative.name.lower()
    is_test = bool(lowered_parts & {"test", "tests", "testing"}) or (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".test.js")
        or name.endswith(".test.ts")
    )
    if (
        is_test
        or relative.suffix.lower() not in _SOURCE_SUFFIXES
        or not candidate.is_file()
    ):
        return None
    return relative.as_posix()


def _annotate_recovery_gate_result(
    result: ClaudeCodeResult,
    validation: _RecoveryGateValidation,
) -> ClaudeCodeResult:
    """Add the state-machine decision into phase metrics and events so result analysis can audit it directly."""

    details = {
        "valid": validation.valid,
        "tool_sequence": list(validation.tool_sequence),
        "target_files": list(validation.target_files),
        "error": validation.error,
        "raw_exit_code": result.exit_code,
    }
    metrics = dict(result.metrics)
    metrics["recovery_gate"] = details
    events = list(result.events)
    events.append(
        {
            "sequence": len(events) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "recovery_gate_validation",
            "details": details,
        }
    )
    # A bounded gate may end with max_turns immediately after a valid final Edit. Once
    # the structured trajectory passes validation, treat that boundary as successful;
    # the raw exit code remains available for auditing.
    exit_code = 0 if validation.valid else result.exit_code
    return replace(
        result,
        exit_code=exit_code,
        events=tuple(events),
        metrics=metrics,
    )


def _should_run_verification(patch: str) -> bool:
    """Decide Verification solely by whether the candidate patch is non-empty, independent of the test plan or results."""

    return bool(patch.strip())


def _build_recovery_handoff(
    result: ClaudeCodeResult,
    *,
    maximum_chars: int,
) -> str:
    """Generate a short handoff from the first round's public trajectory, without copying thinking or tool-return content.

    Recovery is a brand-new model session; if only the issue were resent, it would repeat the first round's locating and exhaust the limited
    turns. This keeps the last few model-visible conclusions and the key parameters of the tools already called, so it can continue directly;
    ``tool_result``, stderr, raw logs, and any hidden reasoning never enter the new prompt.
    """

    visible_texts: list[str] = []
    tool_summaries: list[str] = []
    for event in result.events:
        if event.get("event_type") != "assistant":
            continue
        details = event.get("details")
        if not isinstance(details, Mapping):
            continue
        message = details.get("message")
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue
        for block in content:
            if not isinstance(block, Mapping):
                continue
            block_type = block.get("type")
            if block_type == "text":
                value = block.get("text")
                if isinstance(value, str) and value.strip():
                    visible_texts.append(truncate_output(value.strip(), 1200))
                continue
            if block_type != "tool_use":
                # The explicit whitelist admits only text/tool_use; even if upstream sanitization rules change, thinking
                # and tool_result cannot accidentally enter the Recovery context.
                continue
            name = block.get("name")
            tool_input = block.get("input")
            if not isinstance(name, str) or not isinstance(tool_input, Mapping):
                continue
            summary = _summarize_recovery_tool_call(name, tool_input)
            if summary and summary not in tool_summaries:
                tool_summaries.append(summary)

    terminal_reason = result.metrics.get("terminal_reason", "unknown")
    result_subtype = result.metrics.get("result_subtype", "unknown")
    lines = [
        f"Previous termination: reason={terminal_reason}; subtype={result_subtype}",
    ]
    if visible_texts:
        lines.append("Recent visible conclusions:")
        lines.extend(f"- {text}" for text in visible_texts[-3:])
    if tool_summaries:
        lines.append("Recent inspected resources and tool calls:")
        lines.extend(f"- {summary}" for summary in tool_summaries[-8:])
    if not visible_texts and not tool_summaries:
        lines.append("No usable visible findings or tool calls were recorded.")
    return truncate_output("\n".join(lines), maximum_chars)


def _summarize_recovery_tool_call(name: str, tool_input: Mapping[str, Any]) -> str:
    """Compress the tool arguments allowed for handoff into a single line, avoiding carrying arbitrarily large results."""

    fields_by_tool = {
        "Read": ("file_path", "offset", "limit"),
        "Edit": ("file_path",),
        "Write": ("file_path",),
        "Grep": ("pattern", "path", "glob"),
        "Glob": ("pattern", "path"),
        # Recovery disables Bash, but commands executed in the previous phase can help avoid repeating the locating work.
        "Bash": ("description", "command"),
    }
    fields = fields_by_tool.get(name)
    if fields is None:
        return ""
    parts: list[str] = []
    for field in fields:
        value = tool_input.get(field)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            normalized = " ".join(str(value).split())
            if normalized:
                parts.append(f"{field}={truncate_output(normalized, 300)}")
    return f"{name}: {', '.join(parts)}" if parts else name


def _build_recovery_source_context(
    result: ClaudeCodeResult,
    repository: Path,
    *,
    maximum_chars: int,
) -> str:
    """Extract limited source excerpts from the worktree using the most recent Read arguments, for use by the forced Edit.

    The parent process re-reads the files instead of copying arbitrary ``tool_result`` content, so only existing source inside the repository
    with trusted suffixes is handed off. At most the two most recent read locations are selected, each capped at 80 lines, both providing
    precise old text for the Edit and avoiding stuffing the first round's large output back into context.
    """

    root = repository.resolve()
    requests: list[tuple[Path, int, int]] = []
    for event in result.events:
        if event.get("event_type") != "assistant":
            continue
        details = event.get("details")
        message = details.get("message") if isinstance(details, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue
        for block in content:
            if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                continue
            if block.get("name") != "Read":
                continue
            tool_input = block.get("input")
            if not isinstance(tool_input, Mapping):
                continue
            raw_path = tool_input.get("file_path")
            if not isinstance(raw_path, str):
                continue
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = root / candidate
            candidate = candidate.resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            if (
                candidate.suffix.lower() not in _SOURCE_SUFFIXES
                or not candidate.is_file()
            ):
                continue
            raw_offset = tool_input.get("offset", 1)
            raw_limit = tool_input.get("limit", 80)
            offset = raw_offset if isinstance(raw_offset, int) else 1
            limit = raw_limit if isinstance(raw_limit, int) else 80
            requests.append((candidate, max(1, offset), max(1, min(80, limit))))

    selected: list[tuple[Path, int, int]] = []
    seen: set[tuple[Path, int, int]] = set()
    for request in reversed(requests):
        if request in seen:
            continue
        selected.append(request)
        seen.add(request)
        if len(selected) == 2:
            break

    blocks: list[str] = []
    for path, offset, limit in reversed(selected):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        start = min(offset - 1, len(lines))
        excerpt = lines[start : start + limit]
        numbered = "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(excerpt, start=start + 1)
        )
        blocks.append(f"File: {path.relative_to(root)}\n{numbered}")
    if not blocks:
        return ""
    return truncate_output("\n\n".join(blocks), maximum_chars)


@dataclass(frozen=True, slots=True)
class ScheduledTestExecution:
    """The paired Docker results for the same test argv on the baseline and the candidate patch."""

    label: str
    argv: tuple[str, ...]
    baseline_result: VisibleTestResult | None = None
    baseline_error: str | None = None
    result: VisibleTestResult | None = None
    error: str | None = None

    @property
    def baseline_outcome(self) -> TestOutcome | None:
        """Parse the baseline result; pre-scheduling errors have no output, so return ``None``."""

        if self.baseline_result is None:
            return None
        return parse_test_outcome(self.argv, self.baseline_result)

    @property
    def candidate_outcome(self) -> TestOutcome | None:
        """Parse the candidate result; pre-scheduling errors have no output, so return ``None``."""

        if self.result is None:
            return None
        return parse_test_outcome(self.argv, self.result)

    @property
    def new_failure_signatures(self) -> frozenset[str]:
        """Return the localizable failing tests newly added by the candidate relative to the baseline."""

        baseline = self.baseline_outcome
        candidate = self.candidate_outcome
        if baseline is None or candidate is None:
            return frozenset()
        return candidate.failure_signatures - baseline.failure_signatures

    @property
    def comparison(self) -> str:
        """Combine exit status with failure-signature diffs to identify new regressions on top of existing failures."""

        if (
            self.baseline_error is not None
            or self.error is not None
            or self.baseline_result is None
            or self.result is None
            or not self.baseline_result.evidence_valid
            or not self.result.evidence_valid
        ):
            return "comparison_unavailable"
        baseline = self.baseline_outcome
        candidate = self.candidate_outcome
        assert baseline is not None and candidate is not None
        baseline_passed = baseline.status == "passed"
        patched_passed = candidate.status == "passed"
        if baseline_passed and patched_passed:
            return "both_passed"
        if baseline_passed and not patched_passed:
            return "new_regression"
        if not baseline_passed and patched_passed:
            return "fixed_baseline_failure"
        if self.new_failure_signatures:
            return "new_regression"
        if (
            baseline.status == candidate.status == "failed"
            and baseline.reliable
            and candidate.reliable
        ):
            return "baseline_failure_persists"
        # When both sides are non-zero but the failing tests cannot be reliably extracted, it is better to give no conclusion than to mask the
        # candidate's newly added collection errors or process crashes as "baseline failure persists".
        return "comparison_unavailable"


@dataclass(frozen=True, slots=True)
class ScheduledTestEvidence:
    """A two-command test plan and its auditable Docker execution results."""

    request: TestPlanRequest
    executions: tuple[ScheduledTestExecution, ...] = ()
    task_budget_exhausted: bool = False

    @property
    def ready_for_verification(self) -> bool:
        """Determine whether the two commands form valid comparison evidence; it no longer acts as a hard gate."""

        return (
            self.request.accepted
            and len(self.executions) == len(self.request.commands) == 2
            and all(
                execution.baseline_result is not None
                and execution.baseline_result.evidence_valid
                and execution.result is not None
                and execution.result.evidence_valid
                for execution in self.executions
            )
        )

    @property
    def has_new_regression(self) -> bool:
        """Return whether there is a deterministic new regression where the baseline passes and the candidate fails."""

        return any(
            execution.comparison == "new_regression"
            for execution in self.executions
        )

    def metrics(self) -> dict[str, Any]:
        """Generate metrics that never mis-count textual mentions as container executions."""

        patched_executed = sum(
            item.result is not None
            and item.result.command_started
            and not item.result.cache_hit
            for item in self.executions
        )
        baseline_executed = sum(
            item.baseline_result is not None
            and item.baseline_result.command_started
            and not item.baseline_result.cache_hit
            for item in self.executions
        )
        valid_executions = sum(
            result.evidence_valid and not result.cache_hit
            for item in self.executions
            for result in (item.baseline_result, item.result)
            if result is not None
        )
        infrastructure_errors = sum(
            result.infrastructure_error is not None and not result.cache_hit
            for item in self.executions
            for result in (item.baseline_result, item.result)
            if result is not None
        )
        baseline_failure_signatures = sum(
            len(outcome.failure_signatures)
            for item in self.executions
            if (outcome := item.baseline_outcome) is not None
        )
        candidate_failure_signatures = sum(
            len(outcome.failure_signatures)
            for item in self.executions
            if (outcome := item.candidate_outcome) is not None
        )
        new_failure_signatures = sum(
            len(item.new_failure_signatures) for item in self.executions
        )
        outcome_parser_failures = sum(
            outcome.status == "failed" and not outcome.reliable
            for item in self.executions
            for outcome in (item.baseline_outcome, item.candidate_outcome)
            if outcome is not None
        )
        rejected = self.request.status == "rejected" or any(
            item.baseline_error is not None
            or item.error is not None
            or (
                item.baseline_result is not None
                and not item.baseline_result.command_started
            )
            or (item.result is not None and not item.result.command_started)
            or (
                item.baseline_result is not None
                and item.baseline_result.infrastructure_error is not None
            )
            or (
                item.result is not None
                and item.result.infrastructure_error is not None
            )
            for item in self.executions
        )
        return {
            "agent_turns": 0,
            "tool_calls": 0,
            "host_test_calls": 0,
            "agent_test_command_calls": 0,
            # Legacy metrics keep counting only patched executions, preserving comparability with historical batches; the baseline
            # container count is recorded in a separate field, and only the sum of the two is the actual number of Docker test runs.
            "visible_test_calls": patched_executed,
            "visible_test_requests": int(self.request.requested),
            "visible_test_missing": int(self.request.status == "missing"),
            "visible_test_rejected": int(rejected),
            "visible_test_parent_generated": int(
                self.request.origin == "parent" and self.request.accepted
            ),
            "visible_test_executions": patched_executed,
            # executions keeps the historical semantics of "a container command was actually started"; valid_executions
            # further excludes environment errors such as a missing runner, so new experiments can judge whether the evidence is trustworthy.
            "visible_test_valid_executions": valid_executions,
            "visible_test_infrastructure_errors": infrastructure_errors,
            "visible_test_baseline_failure_signatures": (
                baseline_failure_signatures
            ),
            "visible_test_candidate_failure_signatures": (
                candidate_failure_signatures
            ),
            "visible_test_new_failure_signatures": new_failure_signatures,
            "visible_test_outcome_parser_failures": outcome_parser_failures,
            "test_evidence_available": any(
                item.comparison != "comparison_unavailable"
                for item in self.executions
            ),
            "visible_test_passed": sum(
                item.result is not None and item.result.exit_code == 0
                and item.result.command_started and not item.result.cache_hit
                for item in self.executions
            ),
            "visible_test_baseline_executions": baseline_executed,
            "visible_test_baseline_passed": sum(
                item.baseline_result is not None
                and item.baseline_result.exit_code == 0
                and item.baseline_result.command_started
                and not item.baseline_result.cache_hit
                for item in self.executions
            ),
            "visible_test_comparisons": sum(
                item.comparison != "comparison_unavailable"
                for item in self.executions
            ),
            "visible_test_new_regressions": sum(
                item.comparison == "new_regression" for item in self.executions
            ),
            "visible_test_fixed_baseline_failures": sum(
                item.comparison == "fixed_baseline_failure"
                for item in self.executions
            ),
            "visible_test_unchanged_baseline_failures": sum(
                item.comparison == "baseline_failure_persists"
                for item in self.executions
            ),
            "visible_test_timed_out": any(
                (
                    item.baseline_result is not None
                    and item.baseline_result.command_started
                    and item.baseline_result.timed_out
                    and not item.baseline_result.cache_hit
                )
                or (
                    item.result is not None
                    and item.result.command_started
                    and item.result.timed_out
                    and not item.result.cache_hit
                )
                for item in self.executions
            ),
            "duration_seconds": round(
                sum(
                    result.duration_seconds
                    for item in self.executions
                    for result in (item.baseline_result, item.result)
                    if result is not None
                ),
                6,
            ),
            "cache_hit": any(
                result.cache_hit
                for item in self.executions
                for result in (item.baseline_result, item.result)
                if result is not None
            ),
            "baseline_timed_out": any(
                item.baseline_result is not None
                and item.baseline_result.command_started
                and item.baseline_result.timed_out
                for item in self.executions
            ),
            "candidate_timed_out": any(
                item.result is not None
                and item.result.command_started
                and item.result.timed_out
                for item in self.executions
            ),
            "task_budget_exhausted": self.task_budget_exhausted,
            "timed_out": False,
            "token_usage": {},
        }

    def prompt_text(self) -> str:
        """Compress the real execution evidence into text that can be injected directly into the repair session."""

        if self.request.status == "missing":
            prefix = (
                "The parent scheduler could not generate a repository-adapted test plan"
                if self.request.origin == "parent"
                else "No structured test plan was submitted by the implementation session"
            )
            suffix = f": {self.request.error}" if self.request.error else "."
            return prefix + suffix
        if self.request.status == "rejected":
            return f"The submitted test plan was rejected: {self.request.error}"
        blocks: list[str] = []
        for execution in self.executions:
            comparison = execution.comparison
            block = [
                f"[{execution.label}] argv: {list(execution.argv)!r}",
                f"Comparison: {comparison}",
                "New failure signatures: "
                f"{sorted(execution.new_failure_signatures)!r}",
            ]
            if execution.baseline_error is not None:
                block.append(f"Baseline could not run: {execution.baseline_error}")
            elif execution.baseline_result is not None:
                block.extend(
                    (
                        "Baseline (unmodified image):",
                        f"Command started: {execution.baseline_result.command_started}",
                        f"Exit code: {execution.baseline_result.exit_code}",
                        f"Timed out: {execution.baseline_result.timed_out}",
                        f"Duration seconds: {execution.baseline_result.duration_seconds}",
                        f"Cache hit: {execution.baseline_result.cache_hit}",
                        "Infrastructure error: "
                        f"{execution.baseline_result.infrastructure_error}",
                        "Failure signatures: "
                        f"{sorted(execution.baseline_outcome.failure_signatures)!r}",
                        "Output:",
                        # Each of the two commands has its own baseline/patched output; cap length again per block to avoid
                        # dropping the patched target traceback in the middle when the total evidence is truncated.
                        truncate_output(
                            execution.baseline_result.output,
                            2400,
                        ).rstrip(),
                    )
                )
            if execution.error is not None:
                block.append(f"Patched run could not run: {execution.error}")
            elif execution.result is not None:
                block.extend(
                    (
                        "Patched candidate:",
                        f"Docker image: {execution.result.image}",
                        f"Command started: {execution.result.command_started}",
                        f"Exit code: {execution.result.exit_code}",
                        f"Timed out: {execution.result.timed_out}",
                        f"Duration seconds: {execution.result.duration_seconds}",
                        f"Cache hit: {execution.result.cache_hit}",
                        f"Infrastructure error: {execution.result.infrastructure_error}",
                        "Failure signatures: "
                        f"{sorted(execution.candidate_outcome.failure_signatures)!r}",
                        "Output:",
                        truncate_output(execution.result.output, 2400).rstrip(),
                    )
                )
            blocks.append("\n".join(block))
        return "\n\n".join(blocks)

    def repair_prompt_text(self) -> str:
        """When there is a new regression, hand off only the first certain failure, avoiding dilution of the repair focus by irrelevant tests."""

        for execution in self.executions:
            if execution.comparison == "new_regression":
                focused = ScheduledTestEvidence(
                    request=self.request,
                    executions=(execution,),
                )
                return focused.prompt_text()
        return self.prompt_text()


def _verification_policy(
    evidence: ScheduledTestEvidence,
) -> tuple[str, tuple[str, ...] | None]:
    """Choose the Verification phase name and CLI tool whitelist based on the real regression evidence."""

    if evidence.has_new_regression:
        # A whitelist is more robust than disabling tools one by one: even if Claude Code adds built-in tools, focused mode can
        # still only Read the failing location and Edit existing files, and cannot re-search or delegate sub-agents.
        return "verification_regression_repair", ("Read", "Edit")
    return "verification", None


def _execute_scheduled_test(
    repository: Path,
    *,
    task: SWEbenchTask,
    workspace_base_commit: str,
    sandbox: VisibleTestSandbox,
    request: TestPlanRequest | None = None,
    candidate_patch: str | None = None,
    cache: _TestResultCache | None = None,
    target_timeout_seconds: float | None = None,
    regression_timeout_seconds: float | None = None,
    task_budget: _TaskBudget | None = None,
) -> ScheduledTestEvidence:
    """Run the baseline and patched containers for both commands to form comparable evidence.

    The baseline uses a stable cache key for the empty patch, so a valid result is not re-run between Initial/Final.
    A candidate result is reused only when the patch is exactly the same and it genuinely passed last time; infrastructure errors are never
    written to or read from the cache, and failures and timeouts are also re-run, avoiding turning occasional environment faults into final evidence.
    """

    request = consume_test_plan(repository) if request is None else request
    if not request.accepted:
        return ScheduledTestEvidence(request=request)
    if cache is not None and candidate_patch is None:
        raise ValueError("candidate_patch is required when test cache is enabled")

    patch_digest = hashlib.sha256((candidate_patch or "").encode("utf-8")).hexdigest()
    baseline_digest = hashlib.sha256(b"").hexdigest()
    budget_exhausted = False
    executions: list[ScheduledTestExecution] = []
    for label, argv in request.commands:
        configured_timeout = (
            regression_timeout_seconds
            if label == "regression" and regression_timeout_seconds is not None
            else target_timeout_seconds
        )

        def run_one(*, apply_patch: bool) -> VisibleTestResult:
            """Execute or reuse a single-side result, while tightening the total budget to this subprocess."""

            digest = patch_digest if apply_patch else baseline_digest
            timeout_key = float(configured_timeout or 0.0)
            key: _TestCacheKey = (
                task.instance_id,
                workspace_base_commit,
                tuple(argv),
                digest,
                timeout_key,
            )
            cached = cache.get(key) if cache is not None else None
            can_reuse = cached is not None and cached.evidence_valid and (
                not apply_patch
                or (
                    cached.exit_code == 0
                    and not cached.timed_out
                )
            )
            if can_reuse and cached is not None:
                # duration_seconds represents the actual waiting time of this phase; the historical duration is still kept in the first
                # scheduled phase, so on a cache hit it must be zeroed to avoid double-counting in the summary.
                return replace(cached, duration_seconds=0.0, cache_hit=True)

            timeout_override = configured_timeout
            if task_budget is not None:
                timeout_override = task_budget.limit(
                    configured_timeout
                    if configured_timeout is not None
                    else sandbox.timeout_seconds
                )
            run_options: dict[str, Any] = {
                "instance_id": task.instance_id,
                "base_commit": workspace_base_commit,
                "command": argv,
                "apply_patch": apply_patch,
            }
            # Compatible with test doubles that only implement the old run signature; real scheduling always passes the configured
            # target/regression timeout values explicitly.
            if timeout_override is not None:
                run_options["timeout_seconds"] = timeout_override
            result = sandbox.run(repository, **run_options)
            # Results such as a missing runner have no test semantics; caching them would only make the Final phase keep believing the same
            # fake evidence. Allowing later phases to retry also makes it easy to get real results as soon as the environment is fixed.
            if cache is not None and result.evidence_valid:
                cache[key] = result
            return result

        baseline_result: VisibleTestResult | None = None
        baseline_error: str | None = None
        try:
            baseline_result = run_one(apply_patch=False)
        except TaskBudgetExhausted as error:
            baseline_error = str(error)
            budget_exhausted = True
        except (TestSandboxError, ValueError) as error:
            baseline_error = str(error)

        patched_result: VisibleTestResult | None = None
        patched_error: str | None = None
        if budget_exhausted:
            patched_error = "task wall-clock budget exhausted"
        else:
            try:
                patched_result = run_one(apply_patch=True)
            except TaskBudgetExhausted as error:
                patched_error = str(error)
                budget_exhausted = True
            except (TestSandboxError, ValueError) as error:
                patched_error = str(error)
        executions.append(
            ScheduledTestExecution(
                label=label,
                argv=argv,
                baseline_result=baseline_result,
                baseline_error=baseline_error,
                result=patched_result,
                error=patched_error,
            )
        )
    return ScheduledTestEvidence(
        request=request,
        executions=tuple(executions),
        task_budget_exhausted=budget_exhausted,
    )


def _scheduled_test_result(evidence: ScheduledTestEvidence) -> ClaudeCodeResult:
    """Wrap the scheduler evidence into a phase so the trajectory, logs, and summary keep the same topology."""

    evidence_metrics = evidence.metrics()
    details: dict[str, Any] = {
        "request_status": evidence.request.status,
        "request_origin": evidence.request.origin,
        "commands": [
            {
                "label": execution.label,
                "argv": list(execution.argv),
                "comparison": execution.comparison,
                "new_failure_signatures": sorted(
                    execution.new_failure_signatures
                ),
                "baseline_error": execution.baseline_error,
                "baseline_exit_code": (
                    execution.baseline_result.exit_code
                    if execution.baseline_result is not None
                    else None
                ),
                "baseline_command_started": (
                    execution.baseline_result.command_started
                    if execution.baseline_result is not None
                    else False
                ),
                "baseline_timed_out": (
                    execution.baseline_result.timed_out
                    if execution.baseline_result is not None
                    else False
                ),
                "baseline_duration_seconds": (
                    execution.baseline_result.duration_seconds
                    if execution.baseline_result is not None
                    else 0.0
                ),
                "baseline_cache_hit": (
                    execution.baseline_result.cache_hit
                    if execution.baseline_result is not None
                    else False
                ),
                "baseline_infrastructure_error": (
                    execution.baseline_result.infrastructure_error
                    if execution.baseline_result is not None
                    else None
                ),
                "baseline_failure_signatures": (
                    sorted(execution.baseline_outcome.failure_signatures)
                    if execution.baseline_outcome is not None
                    else []
                ),
                "baseline_outcome_parser": (
                    execution.baseline_outcome.parser
                    if execution.baseline_outcome is not None
                    else None
                ),
                "error": execution.error,
                "exit_code": (
                    execution.result.exit_code
                    if execution.result is not None
                    else None
                ),
                "image": (
                    execution.result.image if execution.result is not None else None
                ),
                "timed_out": (
                    execution.result.timed_out
                    if execution.result is not None
                    else False
                ),
                "command_started": (
                    execution.result.command_started
                    if execution.result is not None
                    else False
                ),
                "candidate_timed_out": (
                    execution.result.timed_out
                    if execution.result is not None
                    else False
                ),
                "candidate_duration_seconds": (
                    execution.result.duration_seconds
                    if execution.result is not None
                    else 0.0
                ),
                "candidate_cache_hit": (
                    execution.result.cache_hit
                    if execution.result is not None
                    else False
                ),
                "candidate_infrastructure_error": (
                    execution.result.infrastructure_error
                    if execution.result is not None
                    else None
                ),
                "candidate_failure_signatures": (
                    sorted(execution.candidate_outcome.failure_signatures)
                    if execution.candidate_outcome is not None
                    else []
                ),
                "candidate_outcome_parser": (
                    execution.candidate_outcome.parser
                    if execution.candidate_outcome is not None
                    else None
                ),
                "duration_seconds": round(
                    sum(
                        result.duration_seconds
                        for result in (
                            execution.baseline_result,
                            execution.result,
                        )
                        if result is not None
                    ),
                    6,
                ),
                "cache_hit": any(
                    result.cache_hit
                    for result in (
                        execution.baseline_result,
                        execution.result,
                    )
                    if result is not None
                ),
                "task_budget_exhausted": evidence.task_budget_exhausted,
            }
            for execution in evidence.executions
        ],
        "error": evidence.request.error,
        "duration_seconds": evidence_metrics["duration_seconds"],
        "cache_hit": evidence_metrics["cache_hit"],
        "baseline_timed_out": evidence_metrics["baseline_timed_out"],
        "candidate_timed_out": evidence_metrics["candidate_timed_out"],
        "task_budget_exhausted": evidence.task_budget_exhausted,
    }
    text = evidence.prompt_text()
    return ClaudeCodeResult(
        # This phase only saves evidence; missing or failing tests no longer block a non-empty patch from entering verification.
        exit_code=0,
        agent_log=text + "\n",
        test_output=text + "\n",
        events=(
            {
                "sequence": 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "visible_test_execution",
                "details": details,
            },
        ),
        timed_out=False,
        metrics=evidence_metrics,
    )


def _task_budget_result(
    phases: Sequence[tuple[str, ClaudeCodeResult]],
) -> ClaudeCodeResult:
    """Convert budget exhaustion into a persistable result, preserving all prior phase evidence."""

    if phases:
        combined = combine_phase_results(phases)
    else:
        combined = ClaudeCodeResult(
            exit_code=124,
            agent_log="",
            test_output="",
            events=(),
            timed_out=True,
            metrics={},
        )
    metrics = dict(combined.metrics)
    metrics.setdefault("duration_seconds", 0.0)
    metrics.setdefault("cache_hit", False)
    metrics.setdefault("baseline_timed_out", False)
    metrics.setdefault("candidate_timed_out", False)
    metrics["task_budget_exhausted"] = True
    metrics["timed_out"] = True
    events = list(combined.events)
    events.append(
        {
            "sequence": len(events) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "task_budget_exhausted",
            "details": {"task_budget_exhausted": True},
        }
    )
    return replace(
        combined,
        exit_code=124,
        agent_log=combined.agent_log + "\nTask wall-clock budget exhausted.\n",
        events=tuple(events),
        timed_out=True,
        metrics=metrics,
    )


def prepare_visible_test_image(
    task: SWEbenchTask,
    config: ExperimentConfig,
    *,
    allow_network_preparation: bool,
) -> str | None:
    """Prepare the test image before the Agent starts; the solving phase must not call this download entry point."""

    if not config.agent.visible_test_sandbox:
        return None
    sandbox = VisibleTestSandbox(
        timeout_seconds=config.agent.visible_test_timeout_seconds,
        max_output_chars=config.agent.max_tool_output_chars,
    )
    return sandbox.ensure_image(
        task.instance_id,
        allow_pull=allow_network_preparation,
    )


def _load_tasks(path: Path) -> SWEbenchLoader:
    """Load task snapshots constrained by the field whitelist according to the file extension."""

    if path.suffix.lower() == ".jsonl":
        return SWEbenchLoader.from_jsonl(path)
    if path.suffix.lower() == ".json":
        return SWEbenchLoader.from_json(path)
    raise ValueError("--tasks must point to a .json or .jsonl file")


def run_claude_task(
    task: SWEbenchTask,
    repository: Path,
    config: ExperimentConfig,
    runs_root: Path,
    *,
    base_url: str,
    workspace_base_commit: str | None = None,
) -> Path:
    """Run the implementation, scheduled-test, and no-Bash repair phases, and persist the complete evidence."""

    patch_base_commit = workspace_base_commit or task.base_commit
    prompt = PromptBuilder.from_file(config.prompt_template_path).build(task)
    runtime_fingerprint = RuntimeFingerprintCollector(
        project_root=config.project_root,
        prompt_path=config.prompt_template_path,
        model_name=config.model.name,
        base_url=base_url,
    ).collect()
    configuration = config.to_metadata()
    configuration["runtime"] = runtime_fingerprint
    if config.agent.visible_test_sandbox:
        # Complete the read-only image precheck before creating the Agent process; a missing image must be handled in
        # the network-allowed preparation stage, not by letting the model try docker pull during the actual solve.
        test_sandbox = VisibleTestSandbox(
            timeout_seconds=config.agent.visible_test_timeout_seconds,
            max_output_chars=config.agent.max_tool_output_chars,
        )
        visible_test_image = test_sandbox.resolve_image(task.instance_id)
        configuration["visible_test_runtime"] = {
            "image": visible_test_image,
            "image_id": test_sandbox.image_digest(visible_test_image),
            "network": "none",
        }
    session = RunManager(runs_root).start(
        task,
        phase=config.experiment.phase,
        agent=config.agent.framework,
        model=config.model.name,
        prompt=prompt,
        configuration=configuration,
        patch_base_commit=patch_base_commit,
    )
    task_budget = _TaskBudget.start(config.agent.task_timeout_seconds)
    test_cache: dict[_TestCacheKey, VisibleTestResult] = {}
    phases: list[tuple[str, ClaudeCodeResult]] = []
    handoff = HandoffLedger()
    candidate_checkpoint = ""
    candidate_checkpoint_syntax_valid = False
    final_syntax_error = False
    implementation_reproduction = IssueReproductionRequest(
        "missing" if config.agent.issue_reproduction_enabled else "disabled"
    )
    latest_reproduction: IssueReproductionEvidence | None = None

    def append_model_phase(name: str, phase_result: ClaudeCodeResult) -> IssueReproductionRequest:
        """Remove control artifacts and save the actual patch before another stage edits it."""

        phases.append((name, phase_result))
        # A model-written control file is never part of the candidate handed to the next stage.
        consume_test_plan(repository)
        reproduction = consume_issue_reproduction(repository)
        if not config.agent.issue_reproduction_enabled:
            # The ablation removes both the instruction and execution while still
            # consuming a stray control file before patch collection.
            reproduction = IssueReproductionRequest("disabled")
        if name == "implementation" and reproduction.accepted:
            session.write_issue_reproduction(reproduction.code)
        handoff.record_model(
            name,
            patch=session.collect_patch(repository),
            exit_code=phase_result.exit_code,
            agent_reported=_build_recovery_handoff(
                phase_result,
                maximum_chars=min(config.agent.max_tool_output_chars, 1200),
            ),
        )
        session.write_handoff(handoff.snapshot())
        return reproduction

    def append_test_phase(name: str, evidence: ScheduledTestEvidence, patch: str) -> None:
        """Attach parent-owned comparisons to the patch revision they actually exercised."""

        phases.append((name, _scheduled_test_result(evidence)))
        handoff.record_test(
            name,
            patch=patch,
            plan_status=evidence.request.status,
            comparisons=[
                {
                    "label": item.label,
                    "comparison": item.comparison,
                    "first_new_failure": next(iter(sorted(item.new_failure_signatures)), ""),
                }
                for item in evidence.executions
            ],
        )
        session.write_handoff(handoff.snapshot())

    def append_syntax_phase(name: str, patch: str) -> SyntaxPreflightEvidence:
        """Record Docker syntax evidence without treating it as a model or test turn."""

        evidence = _syntax_preflight(
            repository,
            task=task,
            base_commit=patch_base_commit,
            patch=patch,
            sandbox=test_sandbox,
            task_budget=task_budget,
        )
        if evidence.status not in {"not_applicable", "unavailable"}:
            phases.append((name, ClaudeCodeResult(
                exit_code=0,
                agent_log="",
                test_output=evidence.prompt_text(),
                events=({
                    "event_type": "syntax_preflight",
                    "details": {"status": evidence.status, "paths": list(evidence.paths)},
                },),
                timed_out=False,
                metrics={
                    "agent_turns": 0,
                    "tool_calls": 0,
                    "syntax_preflight_status": evidence.status,
                    "duration_seconds": evidence.duration_seconds,
                },
            )))
        return evidence

    def append_reproduction_phase(
        name: str, patch: str, request: IssueReproductionRequest,
    ) -> IssueReproductionEvidence:
        """Persist bounded baseline/candidate reproduction evidence for this patch."""

        nonlocal latest_reproduction
        evidence = _execute_issue_reproduction(
            repository,
            task=task,
            base_commit=patch_base_commit,
            request=request,
            sandbox=test_sandbox,
            task_budget=task_budget,
        )
        latest_reproduction = evidence
        if evidence.status not in {"missing", "disabled"}:
            phases.append((name, ClaudeCodeResult(
                exit_code=0,
                agent_log="",
                test_output=evidence.prompt_text(),
                events=({
                    "event_type": "issue_reproduction",
                    "details": {"status": evidence.status, "error": evidence.error},
                },),
                timed_out=False,
                metrics={
                    "agent_turns": 0,
                    "tool_calls": 0,
                    "issue_reproduction_status": evidence.status,
                    "issue_reproduction_error": evidence.error,
                    "baseline_exit_code": (
                        evidence.baseline.exit_code if evidence.baseline is not None else None
                    ),
                    "candidate_exit_code": (
                        evidence.candidate.exit_code if evidence.candidate is not None else None
                    ),
                    "baseline_timed_out": (
                        evidence.baseline.timed_out if evidence.baseline is not None else False
                    ),
                    "candidate_timed_out": (
                        evidence.candidate.timed_out if evidence.candidate is not None else False
                    ),
                    "duration_seconds": sum(
                        result.duration_seconds for result in
                        (evidence.baseline, evidence.candidate) if result is not None
                    ),
                },
            )))
            handoff.record_reproduction(name, patch=patch, status=evidence.status)
            session.write_handoff(handoff.snapshot())
        return evidence
    try:
        if config.agent.verification_turns:
            implementation_turns = (
                config.agent.max_turns
                - config.agent.verification_turns
            )
            # The two model sessions' wall-clock budgets are split hard in proportion to turns; parent test planning and Docker
            # execution consume no model turns, and the final verification absorbs the integer-division remainder.
            implementation_timeout = max(
                1,
                config.agent.timeout_seconds
                * implementation_turns
                // config.agent.max_turns,
            )
            verification_timeout = max(
                1,
                config.agent.timeout_seconds
                - implementation_timeout,
            )
            implementation_prompt = build_implementation_phase_prompt(
                prompt,
                implementation_turns=implementation_turns,
                verification_turns=config.agent.verification_turns,
                max_file_read_lines=config.agent.max_file_read_lines,
                max_tool_output_chars=config.agent.max_tool_output_chars,
                issue_reproduction_enabled=config.agent.issue_reproduction_enabled,
            )
            implementation_result = _runner(
                config,
                turns=implementation_turns,
                timeout_seconds=task_budget.limit(implementation_timeout),
                base_url=base_url,
            ).run(repository, implementation_prompt)
            implementation_reproduction = append_model_phase("implementation", implementation_result)
            task_budget.ensure_remaining()

            # Clean up control files an older model may have left behind, but never adopt their commands. The real test plan is
            # generated deterministically by the parent from the repository layout and the candidate patch.
            consume_test_plan(repository)
            candidate_patch = session.collect_patch(repository)
            if not _should_run_verification(candidate_patch):
                # An empty patch cannot enter Verification, but its reserved turns must
                # not go to waste. A multi-turn Read/Edit gate can repair cross-file
                # defects while the parent audits the complete tool trajectory.
                recovery_handoff = handoff.render_for_prompt()
                source_context = _build_recovery_source_context(
                    implementation_result,
                    repository,
                    maximum_chars=min(
                        config.agent.max_tool_output_chars,
                        5000,
                    ),
                )
                # Keep the three-stage Recovery policy within the original budget. Four
                # gate turns can cover two Read/Edit pairs; three fallback turns remain
                # available before the final three turns reserved for focused repair.
                recovery_gate_turns = min(4, config.agent.verification_turns)
                regression_repair_turns = min(
                    3,
                    max(
                        0,
                        config.agent.verification_turns - recovery_gate_turns,
                    ),
                )
                fallback_turns = max(
                    0,
                    config.agent.verification_turns
                    - recovery_gate_turns
                    - regression_repair_turns,
                )
                recovery_gate_timeout = max(
                    1,
                    verification_timeout
                    * recovery_gate_turns
                    // config.agent.verification_turns,
                )
                fallback_timeout = max(
                    1,
                    verification_timeout
                    * fallback_turns
                    // config.agent.verification_turns,
                )
                regression_repair_timeout = max(
                    1,
                    verification_timeout
                    * regression_repair_turns
                    // config.agent.verification_turns,
                )
                recovery_gate_prompt = build_recovery_read_edit_gate_prompt(
                    prompt,
                    recovery_turns=recovery_gate_turns,
                    implementation_handoff=recovery_handoff,
                    source_context=source_context,
                )
                recovery_gate_result = _runner(
                    config,
                    turns=recovery_gate_turns,
                    timeout_seconds=task_budget.limit(recovery_gate_timeout),
                    base_url=base_url,
                    allow_bash=False,
                    available_tools=("Read", "Edit"),
                ).run(repository, recovery_gate_prompt)
                gate_validation = _validate_recovery_read_edit_gate(
                    recovery_gate_result,
                    repository,
                )
                recovery_gate_result = _annotate_recovery_gate_result(
                    recovery_gate_result,
                    gate_validation,
                )
                append_model_phase("recovery_read_edit_gate", recovery_gate_result)
                task_budget.ensure_remaining()

                recovery_result = recovery_gate_result
                recovered_patch = session.collect_patch(repository)
                gate_source_patch = _existing_source_patch_projection(recovered_patch)
                recovery_patch_valid = (
                    gate_validation.valid
                    and bool(gate_source_patch)
                )
                if not recovery_patch_valid and fallback_turns > 0:
                    fallback_prompt = build_recovery_implementation_prompt(
                        prompt,
                        recovery_turns=fallback_turns,
                        max_file_read_lines=config.agent.max_file_read_lines,
                        max_tool_output_chars=config.agent.max_tool_output_chars,
                        implementation_handoff=handoff.render_for_prompt(),
                    )
                    fallback_result = _runner(
                        config,
                        turns=fallback_turns,
                        timeout_seconds=task_budget.limit(fallback_timeout),
                        base_url=base_url,
                        allow_bash=False,
                    ).run(repository, fallback_prompt)
                    append_model_phase("recovery_fallback", fallback_result)
                    task_budget.ensure_remaining()
                    recovery_result = fallback_result
                    recovered_patch = session.collect_patch(repository)
                    # An invalid gate may already have left a source diff. The fallback
                    # must materially change it rather than inherit an unvalidated patch.
                    fallback_source_patch = _existing_source_patch_projection(
                        recovered_patch
                    )
                    recovery_patch_valid = bool(fallback_source_patch) and (
                        fallback_source_patch != gate_source_patch
                    )

                consume_test_plan(repository)
                recovered_patch = session.collect_patch(repository)
                recovery_syntax = append_syntax_phase(
                    "syntax_preflight_recovery", recovered_patch
                )
                repair_budget_available = regression_repair_turns > 0
                if (
                    not recovery_patch_valid
                    and repair_budget_available
                ):
                    # With no validated source patch there is no regression to repair, so
                    # reuse those turns for one final implementation chance.
                    last_chance_prompt = (
                        build_verification_phase_prompt(
                            prompt,
                            verification_turns=regression_repair_turns,
                            max_file_read_lines=config.agent.max_file_read_lines,
                            max_tool_output_chars=config.agent.max_tool_output_chars,
                            candidate_patch=truncate_output(recovered_patch, config.agent.max_tool_output_chars),
                            syntax_preflight_evidence=recovery_syntax.prompt_text(),
                            focused_syntax_error=True,
                            handoff_text=handoff.render_for_prompt(),
                        )
                        if recovery_syntax.new_syntax_error
                        else build_recovery_implementation_prompt(
                            prompt,
                            recovery_turns=regression_repair_turns,
                            max_file_read_lines=config.agent.max_file_read_lines,
                            max_tool_output_chars=config.agent.max_tool_output_chars,
                            implementation_handoff=handoff.render_for_prompt(),
                            last_chance=True,
                        )
                    )
                    last_chance_result = _runner(
                        config,
                        turns=regression_repair_turns,
                        timeout_seconds=task_budget.limit(
                            regression_repair_timeout
                        ),
                        base_url=base_url,
                        allow_bash=False,
                        **(
                            {"available_tools": ("Read", "Edit")}
                            if recovery_syntax.new_syntax_error else {}
                        ),
                    ).run(repository, last_chance_prompt)
                    append_model_phase("recovery_last_chance", last_chance_result)
                    task_budget.ensure_remaining()
                    recovery_result = last_chance_result
                    repair_budget_available = False
                    consume_test_plan(repository)
                    recovered_patch = session.collect_patch(repository)

                recovery_syntax = append_syntax_phase(
                    "syntax_preflight_recovery_final", recovered_patch
                )
                recovery_reproduction = append_reproduction_phase(
                    "issue_reproduction_recovery", recovered_patch,
                    implementation_reproduction,
                )

                recovery_request = generate_repository_test_plan(
                    repository,
                    repo=task.repo,
                    patch=recovered_patch,
                    problem_statement=task.problem_statement,
                )
                recovery_evidence = _execute_scheduled_test(
                    repository,
                    task=task,
                    workspace_base_commit=patch_base_commit,
                    sandbox=test_sandbox,
                    request=recovery_request,
                    candidate_patch=recovered_patch,
                    cache=test_cache,
                    target_timeout_seconds=(
                        config.agent.visible_test_timeout_seconds
                    ),
                    regression_timeout_seconds=(
                        config.agent.visible_regression_test_timeout_seconds
                    ),
                    task_budget=task_budget,
                )
                append_test_phase("scheduled_test_recovery", recovery_evidence, recovered_patch)
                task_budget.ensure_remaining()

                if (
                    recovery_evidence.has_new_regression
                    or recovery_syntax.new_syntax_error
                    or recovery_reproduction.status in {"new_regression", "still_failing"}
                ) and repair_budget_available:
                    # After Recovery produces a patch it must also close the "test→repair→retest" loop. Here we
                    # reuse the ordinary Verification focused prompt and, at the CLI layer, allow only
                    # Read/Edit so the limited 3 turns do not fall back into broad search.
                    repair_prompt = build_verification_phase_prompt(
                        prompt,
                        verification_turns=regression_repair_turns,
                        max_file_read_lines=config.agent.max_file_read_lines,
                        max_tool_output_chars=config.agent.max_tool_output_chars,
                        candidate_patch=truncate_output(
                            recovered_patch,
                            config.agent.max_tool_output_chars,
                        ),
                        scheduled_test_evidence=truncate_output(
                            recovery_evidence.repair_prompt_text(),
                            config.agent.max_tool_output_chars,
                        ),
                        focused_new_regression=recovery_evidence.has_new_regression,
                        focused_syntax_error=recovery_syntax.new_syntax_error,
                        focused_issue_reproduction=(
                            recovery_reproduction.status in {"new_regression", "still_failing"}
                        ),
                        syntax_preflight_evidence=(
                            recovery_syntax.prompt_text() if recovery_syntax.new_syntax_error else ""
                        ),
                        issue_reproduction_evidence=(
                            recovery_reproduction.prompt_text()
                            if recovery_reproduction.status not in {"missing", "disabled"} else ""
                        ),
                        handoff_text=handoff.render_for_prompt(),
                    )
                    repair_result = _runner(
                        config,
                        turns=regression_repair_turns,
                        timeout_seconds=task_budget.limit(
                            regression_repair_timeout
                        ),
                        base_url=base_url,
                        allow_bash=False,
                        available_tools=("Read", "Edit"),
                    ).run(repository, repair_prompt)
                    append_model_phase("recovery_regression_repair", repair_result)
                    task_budget.ensure_remaining()
                    recovery_result = repair_result

                    consume_test_plan(repository)
                    recovered_patch = session.collect_patch(repository)
                    append_syntax_phase("syntax_preflight_recovery_post_repair", recovered_patch)
                    append_reproduction_phase(
                        "issue_reproduction_recovery_post_repair",
                        recovered_patch,
                        implementation_reproduction,
                    )
                    final_recovery_request = generate_repository_test_plan(
                        repository,
                        repo=task.repo,
                        patch=recovered_patch,
                        problem_statement=task.problem_statement,
                    )
                    final_recovery_evidence = _execute_scheduled_test(
                        repository,
                        task=task,
                        workspace_base_commit=patch_base_commit,
                        sandbox=test_sandbox,
                        request=final_recovery_request,
                        candidate_patch=recovered_patch,
                        cache=test_cache,
                        target_timeout_seconds=(
                            config.agent.visible_test_timeout_seconds
                        ),
                        regression_timeout_seconds=(
                            config.agent.visible_regression_test_timeout_seconds
                        ),
                        task_budget=task_budget,
                    )
                    append_test_phase(
                        "scheduled_test_recovery_final", final_recovery_evidence, recovered_patch
                    )
                    task_budget.ensure_remaining()

                result = combine_phase_results(tuple(phases))
                # The trailing test phase only stores evidence and must not mask the Recovery CLI's own failure;
                # a first-round Implementation failure, however, may be overridden by a successful Recovery.
                result = replace(result, exit_code=recovery_result.exit_code)
            else:
                if _patch_modifies_existing_source(candidate_patch):
                    # Verification can edit freely, but must not erase the last real
                    # source candidate without leaving another source change.
                    candidate_checkpoint = candidate_patch
                    session.write_candidate_checkpoint(candidate_patch)
                candidate_patch_for_prompt = truncate_output(
                    candidate_patch,
                    config.agent.max_tool_output_chars,
                )
                initial_reproduction = append_reproduction_phase(
                    "issue_reproduction_initial", candidate_patch,
                    implementation_reproduction,
                )
                initial_syntax = append_syntax_phase(
                    "syntax_preflight_initial", candidate_patch
                )
                candidate_checkpoint_syntax_valid = initial_syntax.status == "passed"
                parent_request = generate_repository_test_plan(
                    repository,
                    repo=task.repo,
                    patch=candidate_patch,
                    problem_statement=task.problem_statement,
                )
                initial_evidence = _execute_scheduled_test(
                    repository,
                    task=task,
                    workspace_base_commit=patch_base_commit,
                    sandbox=test_sandbox,
                    request=parent_request,
                    candidate_patch=candidate_patch,
                    cache=test_cache,
                    target_timeout_seconds=(
                        config.agent.visible_test_timeout_seconds
                    ),
                    regression_timeout_seconds=(
                        config.agent.visible_regression_test_timeout_seconds
                    ),
                    task_budget=task_budget,
                )
                append_test_phase("scheduled_test_initial", initial_evidence, candidate_patch)
                task_budget.ensure_remaining()
                # Verification's total turns stay unchanged. The main session uses at most 7 turns, and the last 3
                # turns are reserved specifically for regressions Docker only proves after the main session ends.
                post_test_repair_turns = min(
                    3,
                    max(0, config.agent.verification_turns - 1),
                )
                primary_verification_turns = (
                    config.agent.verification_turns - post_test_repair_turns
                )
                primary_verification_timeout = max(
                    1,
                    verification_timeout
                    * primary_verification_turns
                    // config.agent.verification_turns,
                )
                post_test_repair_timeout = max(
                    1,
                    verification_timeout - primary_verification_timeout,
                )
                verification_prompt = build_verification_phase_prompt(
                    prompt,
                    verification_turns=primary_verification_turns,
                    max_file_read_lines=config.agent.max_file_read_lines,
                    max_tool_output_chars=config.agent.max_tool_output_chars,
                    candidate_patch=candidate_patch_for_prompt,
                    scheduled_test_evidence=truncate_output(
                        initial_evidence.repair_prompt_text(),
                        config.agent.max_tool_output_chars,
                    ),
                    focused_new_regression=initial_evidence.has_new_regression,
                    focused_syntax_error=initial_syntax.new_syntax_error,
                    syntax_preflight_evidence=(
                        initial_syntax.prompt_text() if initial_syntax.new_syntax_error else ""
                    ),
                    issue_reproduction_evidence=(
                        initial_reproduction.prompt_text()
                        if initial_reproduction.status not in {"missing", "disabled"} else ""
                    ),
                    handoff_text=handoff.render_for_prompt(),
                )
                verification_phase, verification_available_tools = (
                    _verification_policy(initial_evidence)
                )
                if initial_syntax.new_syntax_error:
                    verification_phase = "verification_syntax_repair"
                    verification_available_tools = ("Read", "Edit")
                # Start verification whenever the candidate patch is non-empty. Bash is disabled at the CLI layer so the model cannot
                # fall back to running tests in the host environment when the parent's tests are missing or fail; when a new regression is found,
                # Grep/Glob are disabled too, confining the session to the failure evidence and the modified files.
                verification_result = _runner(
                    config,
                    turns=primary_verification_turns,
                    timeout_seconds=task_budget.limit(
                        primary_verification_timeout
                    ),
                    base_url=base_url,
                    allow_bash=False,
                    available_tools=verification_available_tools,
                ).run(repository, verification_prompt)
                append_model_phase(verification_phase, verification_result)
                task_budget.ensure_remaining()

                # The verification phase has no authority to replace commands; the parent regenerates them from the final repaired patch,
                # keeping commands consistent with the actual modified paths and independent of model-controlled files.
                consume_test_plan(repository)
                final_patch = session.collect_patch(repository)
                final_reproduction = append_reproduction_phase(
                    "issue_reproduction_final", final_patch,
                    implementation_reproduction,
                )
                final_syntax = append_syntax_phase("syntax_preflight_final", final_patch)
                final_syntax_error = final_syntax.new_syntax_error
                final_request = generate_repository_test_plan(
                    repository,
                    repo=task.repo,
                    patch=final_patch,
                    problem_statement=task.problem_statement,
                )
                final_evidence = _execute_scheduled_test(
                    repository,
                    task=task,
                    workspace_base_commit=patch_base_commit,
                    sandbox=test_sandbox,
                    request=final_request,
                    candidate_patch=final_patch,
                    cache=test_cache,
                    target_timeout_seconds=(
                        config.agent.visible_test_timeout_seconds
                    ),
                    regression_timeout_seconds=(
                        config.agent.visible_regression_test_timeout_seconds
                    ),
                    task_budget=task_budget,
                )
                append_test_phase("scheduled_test_final", final_evidence, final_patch)
                task_budget.ensure_remaining()

                final_model_result = verification_result
                if (
                    final_evidence.has_new_regression
                    or final_syntax.new_syntax_error
                    or final_reproduction.status in {"new_regression", "still_failing"}
                ) and post_test_repair_turns > 0:
                    # Reserve these turns for a proven test regression, syntax error,
                    # or a bounded issue assertion that still fails on the candidate;
                    # missing/incomparable evidence alone never triggers blind repair.
                    post_repair_prompt = build_verification_phase_prompt(
                        prompt,
                        verification_turns=post_test_repair_turns,
                        max_file_read_lines=config.agent.max_file_read_lines,
                        max_tool_output_chars=config.agent.max_tool_output_chars,
                        candidate_patch=truncate_output(
                            final_patch,
                            config.agent.max_tool_output_chars,
                        ),
                        scheduled_test_evidence=truncate_output(
                            final_evidence.repair_prompt_text(),
                            config.agent.max_tool_output_chars,
                        ),
                        focused_new_regression=final_evidence.has_new_regression,
                        focused_syntax_error=final_syntax.new_syntax_error,
                        focused_issue_reproduction=(
                            final_reproduction.status in {"new_regression", "still_failing"}
                        ),
                        syntax_preflight_evidence=(
                            final_syntax.prompt_text() if final_syntax.new_syntax_error else ""
                        ),
                        issue_reproduction_evidence=(
                            final_reproduction.prompt_text()
                            if final_reproduction.status not in {"missing", "disabled"} else ""
                        ),
                        handoff_text=handoff.render_for_prompt(),
                    )
                    post_repair_result = _runner(
                        config,
                        turns=post_test_repair_turns,
                        timeout_seconds=task_budget.limit(
                            post_test_repair_timeout
                        ),
                        base_url=base_url,
                        allow_bash=False,
                        available_tools=("Read", "Edit"),
                    ).run(repository, post_repair_prompt)
                    append_model_phase("verification_post_test_repair", post_repair_result)
                    task_budget.ensure_remaining()
                    final_model_result = post_repair_result

                    consume_test_plan(repository)
                    post_repair_patch = session.collect_patch(repository)
                    append_reproduction_phase(
                        "issue_reproduction_post_repair", post_repair_patch,
                        implementation_reproduction,
                    )
                    post_repair_syntax = append_syntax_phase("syntax_preflight_post_repair", post_repair_patch)
                    final_syntax_error = post_repair_syntax.new_syntax_error
                    post_repair_request = generate_repository_test_plan(
                        repository,
                        repo=task.repo,
                        patch=post_repair_patch,
                        problem_statement=task.problem_statement,
                    )
                    post_repair_evidence = _execute_scheduled_test(
                        repository,
                        task=task,
                        workspace_base_commit=patch_base_commit,
                        sandbox=test_sandbox,
                        request=post_repair_request,
                        candidate_patch=post_repair_patch,
                        cache=test_cache,
                        target_timeout_seconds=(
                            config.agent.visible_test_timeout_seconds
                        ),
                        regression_timeout_seconds=(
                            config.agent.visible_regression_test_timeout_seconds
                        ),
                        task_budget=task_budget,
                    )
                    append_test_phase(
                        "scheduled_test_post_repair", post_repair_evidence, post_repair_patch
                    )
                    task_budget.ensure_remaining()

                result = combine_phase_results(tuple(phases))
                # Missing tests, launch failures, and non-zero exit codes all only record evidence; the model session state is still
                # decided by the last model repair phase that actually ran, and correctness is still adjudicated by the official harness.
                result = replace(result, exit_code=final_model_result.exit_code)
        else:
            # The frozen official baseline config keeps using the original single-session path so the experiment protocol
            # corresponding to a historical fingerprint is not silently rewritten by later architecture work.
            result = _runner(
                config,
                turns=config.agent.max_turns,
                timeout_seconds=task_budget.limit(config.agent.timeout_seconds),
                base_url=base_url,
            ).run(repository, prompt)
            implementation_reproduction = append_model_phase("implementation", result)
            task_budget.ensure_remaining()
        patch = session.collect_patch(repository)
        patch, result = _select_source_checkpoint(
            patch, candidate_checkpoint, result, handoff, session,
            restore_broken_syntax=(candidate_checkpoint_syntax_valid and final_syntax_error),
        )
        result = _apply_patch_gate(result, patch)
    except TaskBudgetExhausted:
        # Total-budget exhaustion is an expected resource boundary and must not take the infrastructure-error branch; keeping the existing patch
        # lets the official harness still evaluate the candidate fix completed before the deadline.
        patch = session.collect_patch(repository)
        patch, result = _select_source_checkpoint(
            patch, candidate_checkpoint, _task_budget_result(phases), handoff, session
        )
        result = _apply_patch_gate(result, patch)
    except Exception as error:
        # Whether the Agent or the patch collection fails, the metadata's running state must be finalized,
        # otherwise later analysis cannot distinguish "still running" from "exited on an infrastructure error".
        try:
            partial_patch = session.collect_patch(repository)
        except Exception as patch_error:
            partial_patch = ""
            patch_note = f"; patch collection failed: {patch_error}"
        else:
            patch_note = ""
        session.finalize(
            exit_code=1,
            agent_log=f"{type(error).__name__}: {error}{patch_note}\n",
            test_output="",
            events=[
                {
                    "sequence": 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "agent_error",
                    "details": {
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                }
            ],
            patch=partial_patch,
        )
        raise

    # A missing submission is still a measured outcome. Phase-only metrics omit
    # missing requests entirely, which previously made uptake impossible to audit.
    final_metrics = dict(result.metrics)
    final_metrics["issue_reproduction"] = {
        "submission_status": implementation_reproduction.status,
        "rejection_reason": implementation_reproduction.error,
        "outcome": latest_reproduction.status if latest_reproduction is not None else None,
    }
    session.finalize(
        exit_code=result.exit_code,
        agent_log=result.agent_log,
        test_output=result.test_output,
        events=result.events,
        patch=patch,
        metrics=final_metrics,
    )
    return session.path


def build_parser() -> argparse.ArgumentParser:
    """Declare single-task run arguments; the run ID is required explicitly to prevent overwriting or accidental result reuse."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument(
        "--allow-network-preparation",
        action="store_true",
        help="Allow Git clone/fetch before the offline Agent subprocess starts.",
    )
    return parser


def main() -> int:
    """Validate config, prepare the worktree, and launch an Agent that only connects to the local Ollama."""

    arguments = build_parser().parse_args()
    if not _SAFE_RUN_ID.fullmatch(arguments.run_id):
        raise ValueError("--run-id may contain only letters, digits, dot, underscore and dash")

    config = ExperimentConfig.load(arguments.config)
    if config.experiment.phase == "evaluation":
        config.require_frozen()
    task = _load_tasks(arguments.tasks).get(arguments.instance_id)
    manager = RepositoryManager(
        config.project_root / config.storage.repository_cache,
        config.project_root / config.storage.workspaces / arguments.run_id,
    )
    prepared = manager.prepare(
        task,
        allow_network=arguments.allow_network_preparation,
    )
    prepare_visible_test_image(
        task,
        config,
        allow_network_preparation=arguments.allow_network_preparation,
    )
    run_path = run_claude_task(
        task,
        prepared.path,
        config,
        config.project_root / config.storage.runs / arguments.run_id,
        base_url=arguments.base_url,
        workspace_base_commit=prepared.workspace_base_commit,
    )
    print(run_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
