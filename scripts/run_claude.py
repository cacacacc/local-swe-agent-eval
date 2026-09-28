"""使用本地 Ollama 驱动 Claude Code，完成一道已准备的 SWE-bench 任务。

输入任务文件必须是经过 ``SWEbenchLoader`` 安全投影的 JSON/JSONL，不能直接把
包含 gold patch 或 test patch 的原始数据记录传给本脚本。
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
from agent.prompt_builder import (
    PromptBuilder,
    build_implementation_phase_prompt,
    build_recovery_edit_gate_prompt,
    build_recovery_implementation_prompt,
    build_verification_phase_prompt,
)
from agent.test_plan import (
    TestPlanRequest,
    consume_test_plan,
    generate_repository_test_plan,
)
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
from tracking.run_manager import RunManager


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
    """单题总墙钟预算耗尽；调用方仍需保存已经形成的 patch 和轨迹。"""


@dataclass(frozen=True, slots=True)
class _TaskBudget:
    """用单调时钟把分散的模型与 Docker 超时约束在同一截止时间内。"""

    deadline_monotonic: float | None

    @classmethod
    def start(cls, timeout_seconds: int) -> "_TaskBudget":
        """从当前时刻启动预算；0 是旧配置使用的兼容关闭值。"""

        deadline = (
            time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
        )
        return cls(deadline_monotonic=deadline)

    def limit(self, requested_seconds: float) -> float:
        """返回不越过总截止时间的子阶段预算，耗尽时立即中止调度。"""

        if requested_seconds <= 0:
            raise ValueError("requested timeout must be positive")
        if self.deadline_monotonic is None:
            return requested_seconds
        remaining = self.deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise TaskBudgetExhausted("task wall-clock budget exhausted")
        # subprocess 接受浮点秒；保留毫秒级余量可避免不足一秒时人为再放宽一秒。
        return min(requested_seconds, max(0.001, remaining))

    def ensure_remaining(self) -> None:
        """阶段结束后阻止下一个模型或 Docker 进程越过单题截止时间。"""

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
) -> ClaudeCodeRunner:
    """按阶段预算构造 Claude Code，并传递 CLI 工具黑白名单。"""

    return ClaudeCodeRunner(
        model=config.model.name,
        timeout_seconds=timeout_seconds,
        max_turns=turns,
        context_length=config.model.context_length,
        max_output_tokens=config.model.max_output_tokens,
        base_url=base_url,
        allow_bash=allow_bash,
        available_tools=available_tools,
    )


def _apply_patch_gate(result: ClaudeCodeResult, patch: str) -> ClaudeCodeResult:
    """把空补丁从“正常完成”改为明确失败，并记录测试证据是否存在。

    该门禁不把“运行过测试”当作成功，因为部分仓库在宿主环境没有依赖；真正的
    resolved 状态仍只由官方 harness 决定。它只消除模型通用回复造成的假完成。
    """

    patch_generated = bool(patch.strip())
    existing_source_modified = _patch_modifies_existing_source(patch)
    metrics = dict(result.metrics)
    visible_executions = int(metrics.get("visible_test_executions", 0))
    host_attempts = int(metrics.get("host_test_calls", 0))
    metrics["patch_gate"] = {
        "patch_generated": patch_generated,
        "existing_source_modified": existing_source_modified,
        # 宿主测试没有使用 SWE-bench instance 环境，永远不能满足测试门禁。
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
        # 2 表示运行器的交付物门禁失败，区别于 Claude CLI 自身的退出码 1。
        exit_code = 2
    return replace(result, exit_code=exit_code, events=tuple(events), metrics=metrics)


def _patch_modifies_existing_source(patch: str) -> bool:
    """判断补丁是否修改至少一个既有的非测试源码文件。

    该门禁专门拦截只留下复现脚本、临时文本或新增测试目录的探索结果。已有测试
    文件的修改也不算产品修复，避免模型通过改测试绕过交付要求。
    """

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
        # 文档、日志和任意既有临时文件同样不能满足“修复产品源码”的要求；
        # 显式后缀白名单适配 SWE-bench 中常见的 Python/C/前端与模板源码。
        if not is_test and path.suffix.lower() in _SOURCE_SUFFIXES and "@@" in section:
            return True
    return False


def _should_run_verification(patch: str) -> bool:
    """只以候选 patch 是否非空决定 Verification，不依赖测试计划或结果。"""

    return bool(patch.strip())


def _build_recovery_handoff(
    result: ClaudeCodeResult,
    *,
    maximum_chars: int,
) -> str:
    """从首轮公开轨迹生成短交接，不复制 thinking 和工具返回内容。

    Recovery 是全新模型会话，若只重新发送 issue，它会重复首轮定位并耗尽有限
    turns。这里保留最后几段模型可见结论及已调用工具的关键参数，让它能直接继续；
    ``tool_result``、stderr、原始日志和任意隐藏推理均不进入新 prompt。
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
                # 显式白名单只接纳 text/tool_use；即便上游清洗规则改变，thinking
                # 和 tool_result 也不会意外进入 Recovery 上下文。
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
    """把允许交接的工具参数压缩为单行，避免携带任意大段结果。"""

    fields_by_tool = {
        "Read": ("file_path", "offset", "limit"),
        "Edit": ("file_path",),
        "Write": ("file_path",),
        "Grep": ("pattern", "path", "glob"),
        "Glob": ("pattern", "path"),
        # Recovery 禁用 Bash，但上一阶段执行过的命令可以帮助避免重复定位。
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
    """按最近 Read 参数从 worktree 提取受限源码片段，供强制 Edit 使用。

    父进程重新读取文件，而不是复制任意 ``tool_result``，因此只会交接仓库内、
    后缀受信任的现有源码。最多选择两个最近读取位置且每处不超过 80 行，既为
    Edit 提供精确 old text，也避免把首轮的大段输出重新塞进上下文。
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
    """同一测试 argv 在基线和候选 patch 上的成对 Docker 结果。"""

    label: str
    argv: tuple[str, ...]
    baseline_result: VisibleTestResult | None = None
    baseline_error: str | None = None
    result: VisibleTestResult | None = None
    error: str | None = None

    @property
    def comparison(self) -> str:
        """按 pass/fail 转换分类；只有 baseline pass → patched fail 算新回归。"""

        if (
            self.baseline_error is not None
            or self.error is not None
            or self.baseline_result is None
            or self.result is None
            or not self.baseline_result.command_started
            or not self.result.command_started
        ):
            return "comparison_unavailable"
        baseline_passed = self.baseline_result.exit_code == 0
        patched_passed = self.result.exit_code == 0
        if baseline_passed and patched_passed:
            return "both_passed"
        if baseline_passed and not patched_passed:
            return "new_regression"
        if not baseline_passed and patched_passed:
            return "fixed_baseline_failure"
        return "baseline_failure_persists"


@dataclass(frozen=True, slots=True)
class ScheduledTestEvidence:
    """一份双命令测试计划及其可审计 Docker 执行结果。"""

    request: TestPlanRequest
    executions: tuple[ScheduledTestExecution, ...] = ()
    task_budget_exhausted: bool = False

    @property
    def ready_for_verification(self) -> bool:
        """兼容指标：判断两条命令是否真实进入 Docker，不再充当硬门禁。"""

        return (
            self.request.accepted
            and len(self.executions) == len(self.request.commands) == 2
            and all(
                execution.result is not None and execution.result.command_started
                for execution in self.executions
            )
        )

    @property
    def has_new_regression(self) -> bool:
        """返回是否存在基线通过、候选失败的确定性新增回归。"""

        return any(
            execution.comparison == "new_regression"
            for execution in self.executions
        )

    def metrics(self) -> dict[str, Any]:
        """生成不会把文本提及误算为容器执行的指标。"""

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
        rejected = self.request.status == "rejected" or any(
            item.baseline_error is not None
            or item.error is not None
            or (
                item.baseline_result is not None
                and not item.baseline_result.command_started
            )
            or (item.result is not None and not item.result.command_started)
            for item in self.executions
        )
        return {
            "agent_turns": 0,
            "tool_calls": 0,
            "host_test_calls": 0,
            "agent_test_command_calls": 0,
            # 旧指标继续只计算 patched 执行，保证与历史批次可比较；baseline
            # 容器次数由独立字段记录，二者相加才是实际 Docker 测试次数。
            "visible_test_calls": patched_executed,
            "visible_test_requests": int(self.request.requested),
            "visible_test_missing": int(self.request.status == "missing"),
            "visible_test_rejected": int(rejected),
            "visible_test_parent_generated": int(
                self.request.origin == "parent" and self.request.accepted
            ),
            "visible_test_executions": patched_executed,
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
        """把真实执行证据压缩成可直接注入修复会话的文本。"""

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
                        "Output:",
                        # 两条命令各有 baseline/patched 输出；再次按单块限长，避免
                        # 总证据截断时恰好丢掉位于中间的 patched target traceback。
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
                        "Output:",
                        truncate_output(execution.result.output, 2400).rstrip(),
                    )
                )
            blocks.append("\n".join(block))
        return "\n\n".join(blocks)

    def repair_prompt_text(self) -> str:
        """新增回归时只交接第一条确定失败，避免无关测试稀释修复注意力。"""

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
    """根据真实回归证据选择 Verification phase 名称和 CLI 工具白名单。"""

    if evidence.has_new_regression:
        # 白名单比逐个禁用更稳健：即使 Claude Code 新增内置工具，聚焦模式仍然
        # 只能 Read 失败位置并 Edit 既有文件，不能重新搜索或委派子代理。
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
    """对双命令分别运行基线与 patched 容器，形成可比较证据。

    baseline 使用空 patch 的稳定缓存键，因此 Initial/Final 不会重复运行。候选
    结果只有在 patch 完全相同且上次真实通过时才复用；失败和超时仍会重跑，避免
    把一次偶发 Docker 故障固化成最终证据。
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
            """执行或复用单侧结果，同时把总预算收紧到本次子进程。"""

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
            can_reuse = cached is not None and (
                not apply_patch
                or (
                    cached.command_started
                    and cached.exit_code == 0
                    and not cached.timed_out
                )
            )
            if can_reuse and cached is not None:
                # duration_seconds 表示本阶段实际等待时间；历史耗时仍保留在首次
                # scheduled phase，因此命中缓存时必须归零，避免汇总重复计费。
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
            # 兼容只实现旧 run 签名的测试替身；真实调度传入配置后始终显式使用
            # target/regression 各自的超时值。
            if timeout_override is not None:
                run_options["timeout_seconds"] = timeout_override
            result = sandbox.run(repository, **run_options)
            if cache is not None:
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
    """把调度器证据包装成 phase，使 trajectory、日志和汇总保持同一拓扑。"""

    evidence_metrics = evidence.metrics()
    details: dict[str, Any] = {
        "request_status": evidence.request.status,
        "request_origin": evidence.request.origin,
        "commands": [
            {
                "label": execution.label,
                "argv": list(execution.argv),
                "comparison": execution.comparison,
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
        # 此 phase 只负责保存证据；测试缺失或失败不再阻断非空补丁进入验证。
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
    """把预算耗尽转换为可持久化结果，并保留此前所有阶段证据。"""

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
    """在 Agent 启动前准备测试镜像；求解阶段不得调用此下载入口。"""

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
    """按扩展名加载经过字段白名单约束的任务快照。"""

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
    """执行实现、调度测试与无 Bash 修复阶段，并持久化完整证据。"""

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
        # 在创建 Agent 进程前完成只读镜像预检；缺失镜像必须回到允许联网的环境
        # 准备阶段处理，不能让模型在正式求解期间尝试 docker pull。
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
    try:
        if config.agent.verification_turns:
            implementation_turns = (
                config.agent.max_turns
                - config.agent.verification_turns
            )
            # 两个模型会话的墙钟预算按 turns 比例硬切分；父进程测试规划和 Docker
            # 执行不消耗模型 turns，最后的 verification 吸收整数除法余数。
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
            )
            implementation_result = _runner(
                config,
                turns=implementation_turns,
                timeout_seconds=task_budget.limit(implementation_timeout),
                base_url=base_url,
            ).run(repository, implementation_prompt)
            phases.append(("implementation", implementation_result))
            task_budget.ensure_remaining()

            # 清理旧模型可能遗留的控制文件但绝不采用其中命令。实际测试计划只由
            # 父进程根据仓库布局与候选 patch 确定性生成。
            consume_test_plan(repository)
            candidate_patch = session.collect_patch(repository)
            if not _should_run_verification(candidate_patch):
                # 空 patch 无法进入 Verification，但其预留 turns 不能白白浪费。
                # 先用两 turns 的 Read→Edit gate：第一 turn 满足 Claude Code 的
                # 编辑前置读取，第二 turn 立即修改。只有未形成产品源码 patch 时，
                # 才把剩余预算交给可探索的 fallback。
                recovery_handoff = _build_recovery_handoff(
                    implementation_result,
                    maximum_chars=min(
                        config.agent.max_tool_output_chars,
                        6000,
                    ),
                )
                source_context = _build_recovery_source_context(
                    implementation_result,
                    repository,
                    maximum_chars=min(
                        config.agent.max_tool_output_chars,
                        5000,
                    ),
                )
                edit_gate_turns = min(2, config.agent.verification_turns)
                fallback_turns = config.agent.verification_turns - edit_gate_turns
                edit_gate_timeout = max(
                    1,
                    verification_timeout
                    * edit_gate_turns
                    // config.agent.verification_turns,
                )
                edit_gate_prompt = build_recovery_edit_gate_prompt(
                    prompt,
                    recovery_turns=edit_gate_turns,
                    implementation_handoff=recovery_handoff,
                    source_context=source_context,
                )
                edit_gate_result = _runner(
                    config,
                    turns=edit_gate_turns,
                    timeout_seconds=task_budget.limit(edit_gate_timeout),
                    base_url=base_url,
                    allow_bash=False,
                    available_tools=("Read", "Edit"),
                ).run(repository, edit_gate_prompt)
                phases.append(("recovery_edit_gate", edit_gate_result))
                task_budget.ensure_remaining()

                recovered_patch = session.collect_patch(repository)
                recovery_result = edit_gate_result
                if (
                    not _patch_modifies_existing_source(recovered_patch)
                    and fallback_turns > 0
                ):
                    fallback_prompt = build_recovery_implementation_prompt(
                        prompt,
                        recovery_turns=fallback_turns,
                        max_file_read_lines=config.agent.max_file_read_lines,
                        max_tool_output_chars=config.agent.max_tool_output_chars,
                        implementation_handoff=recovery_handoff,
                    )
                    fallback_result = _runner(
                        config,
                        turns=fallback_turns,
                        timeout_seconds=task_budget.limit(
                            max(
                                1,
                                verification_timeout - edit_gate_timeout,
                            )
                        ),
                        base_url=base_url,
                        allow_bash=False,
                    ).run(repository, fallback_prompt)
                    phases.append(("recovery_fallback", fallback_result))
                    task_budget.ensure_remaining()
                    recovery_result = fallback_result

                consume_test_plan(repository)
                recovered_patch = session.collect_patch(repository)
                recovery_request = generate_repository_test_plan(
                    repository,
                    repo=task.repo,
                    patch=recovered_patch,
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
                phases.append(
                    (
                        "scheduled_test_recovery",
                        _scheduled_test_result(recovery_evidence),
                    )
                )
                task_budget.ensure_remaining()
                result = combine_phase_results(tuple(phases))
                # 末尾测试 phase 只保存证据，不能掩盖 Recovery CLI 自身的失败；
                # 首轮 Implementation 的失败则允许被成功 Recovery 覆盖。
                result = replace(result, exit_code=recovery_result.exit_code)
            else:
                candidate_patch_for_prompt = truncate_output(
                    candidate_patch,
                    config.agent.max_tool_output_chars,
                )
                parent_request = generate_repository_test_plan(
                    repository,
                    repo=task.repo,
                    patch=candidate_patch,
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
                phases.append(
                    (
                        "scheduled_test_initial",
                        _scheduled_test_result(initial_evidence),
                    )
                )
                task_budget.ensure_remaining()
                verification_prompt = build_verification_phase_prompt(
                    prompt,
                    verification_turns=config.agent.verification_turns,
                    max_file_read_lines=config.agent.max_file_read_lines,
                    max_tool_output_chars=config.agent.max_tool_output_chars,
                    candidate_patch=candidate_patch_for_prompt,
                    scheduled_test_evidence=truncate_output(
                        initial_evidence.repair_prompt_text(),
                        config.agent.max_tool_output_chars,
                    ),
                    focused_new_regression=initial_evidence.has_new_regression,
                )
                verification_phase, verification_available_tools = (
                    _verification_policy(initial_evidence)
                )
                # 只要候选 patch 非空就启动验证。Bash 在 CLI 层禁用，避免模型因
                # 父进程测试缺失或失败而回退到宿主环境自行执行；发现新增回归时
                # 同时禁用 Grep/Glob，把会话限制在失败证据和已修改文件内。
                verification_result = _runner(
                    config,
                    turns=config.agent.verification_turns,
                    timeout_seconds=task_budget.limit(verification_timeout),
                    base_url=base_url,
                    allow_bash=False,
                    available_tools=verification_available_tools,
                ).run(repository, verification_prompt)
                phases.append((verification_phase, verification_result))
                task_budget.ensure_remaining()

                # 验证阶段无权替换命令；父进程根据修复后的最终 patch 重新生成，
                # 保证命令始终与实际修改路径一致且不依赖模型控制文件。
                consume_test_plan(repository)
                final_patch = session.collect_patch(repository)
                final_request = generate_repository_test_plan(
                    repository,
                    repo=task.repo,
                    patch=final_patch,
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
                phases.append(
                    ("scheduled_test_final", _scheduled_test_result(final_evidence))
                )
                task_budget.ensure_remaining()
                result = combine_phase_results(tuple(phases))
                # 测试缺失、启动失败和非零退出码都只记录证据；模型会话状态仍由
                # verification 决定，正确性最终只由官方 SWE-bench harness 裁决。
                result = replace(result, exit_code=verification_result.exit_code)
        else:
            # 已冻结的正式基线配置继续走原来的单会话路径，保证历史 fingerprint
            # 对应的实验协议不被后续架构开发悄悄改写。
            result = _runner(
                config,
                turns=config.agent.max_turns,
                timeout_seconds=task_budget.limit(config.agent.timeout_seconds),
                base_url=base_url,
            ).run(repository, prompt)
            phases.append(("implementation", result))
            task_budget.ensure_remaining()
        patch = session.collect_patch(repository)
        result = _apply_patch_gate(result, patch)
    except TaskBudgetExhausted:
        # 总预算耗尽是预期的资源边界，不应走基础设施异常分支；保留已有 patch，
        # 让官方 harness 仍能评价截止时间前已经完成的候选修复。
        patch = session.collect_patch(repository)
        result = _apply_patch_gate(_task_budget_result(phases), patch)
    except Exception as error:
        # 无论 Agent 还是 patch 收集失败，都要终结 metadata 的 running 状态，
        # 否则后续分析无法区分“仍在运行”和“基础设施异常退出”。
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

    session.finalize(
        exit_code=result.exit_code,
        agent_log=result.agent_log,
        test_output=result.test_output,
        events=result.events,
        patch=patch,
        metrics=result.metrics,
    )
    return session.path


def build_parser() -> argparse.ArgumentParser:
    """声明单题运行参数；run ID 强制显式给出以防覆盖或结果误复用。"""

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
    """校验配置、准备 worktree，并启动只连接本机 Ollama 的 Agent。"""

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
