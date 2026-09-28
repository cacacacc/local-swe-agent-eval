"""使用固定模板为每道 SWE-bench 任务生成可审计 Prompt。

模板在构造时一次性验证，渲染时只接收 ``SWEbenchTask`` 的安全字段，确保不同
任务之间只有题目数据变化，而实验指令保持一致。
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from benchmark.task import SWEbenchTask


class PromptTemplateError(ValueError):
    """当 Prompt 模板为空、不可读或缺少必要占位符时抛出。"""


class PromptBuilder:
    """加载并验证模板，只使用批准的任务字段完成渲染。"""

    REQUIRED_PLACEHOLDERS = (
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
    )

    def __init__(self, template_text: str) -> None:
        """规范化换行符并确认四个任务字段均有显式占位符。"""

        # 统一成 LF，避免 Windows/WSL 行尾差异改变 prompt hash。
        normalized = template_text.replace("\r\n", "\n").replace("\r", "\n")
        if not normalized.strip():
            raise PromptTemplateError("prompt template cannot be empty")
        missing = [
            name
            for name in self.REQUIRED_PLACEHOLDERS
            if f"${{{name}}}" not in normalized
        ]
        if missing:
            raise PromptTemplateError(
                f"prompt template is missing placeholder(s): {', '.join(missing)}"
            )
        self._template = Template(normalized)

    @classmethod
    def from_file(cls, path: Path | str) -> "PromptBuilder":
        """从 UTF-8 文本文件读取模板，并把文件系统错误转换成领域错误。"""

        source = Path(path)
        try:
            return cls(source.read_text(encoding="utf-8"))
        except OSError as error:
            raise PromptTemplateError(f"cannot read prompt template {source}: {error}") from error

    def build(self, task: SWEbenchTask) -> str:
        """渲染单个任务，并保证结果恰好以一个换行符结束。"""

        # ``to_agent_payload`` 是唯一的数据入口，因此评测专用字段无法被替换进模板。
        rendered = self._template.substitute(task.to_agent_payload())
        return f"{rendered.rstrip()}\n"


def build_implementation_phase_prompt(
    base_prompt: str,
    *,
    implementation_turns: int,
    verification_turns: int,
    max_file_read_lines: int,
    max_tool_output_chars: int,
) -> str:
    """为实现阶段追加补丁交付点和上下文预算护栏。

    Claude Code 暂不提供可靠的逐次 Read/Bash 输出硬上限，因此这里把限制写成
    可审计的阶段协议。真实测试统一由父进程在候选补丁产生后调度，Implementation
    不再承担容易受宿主环境误导的前置测试。
    """

    return (
        f"{base_prompt.rstrip()}\n\n"
        "Current phase: implementation\n"
        f"- You have at most {implementation_turns} turns in this phase.\n"
        "- Produce a non-empty candidate patch before this phase ends.\n"
        "- Do not create a Git commit; leave the candidate change in the working tree.\n"
        "- Start with rg or another targeted search; do not dump whole large files.\n"
        f"- Read at most {max_file_read_lines} source lines in one tool call.\n"
        f"- Keep each command output below about {max_tool_output_chars} characters.\n"
        "- Before assuming how an internal API works, find an existing repository usage.\n"
        "- Do not run pytest, tox, project test scripts, or package installers directly "
        "on the host. The parent scheduler owns test execution after this phase.\n"
        "- Do not create `.agent-test-plan.json`; the parent scheduler selects visible "
        "test commands deterministically from the repository and candidate patch.\n"
        f"- A fresh verification session owns the final {verification_turns} turns, so "
        "leave the working tree with your best concrete patch even if local dependencies "
        "prevent tests from running.\n"
    )


def build_verification_phase_prompt(
    base_prompt: str,
    *,
    verification_turns: int,
    max_file_read_lines: int,
    max_tool_output_chars: int,
    candidate_patch: str = "",
    scheduled_test_evidence: str = "",
    focused_new_regression: bool = False,
) -> str:
    """注入候选补丁和测试证据，按是否产生新回归构造修复 Prompt。"""
    patch_block = (
        "Candidate patch collected relative to the task base commit\n"
        "<candidate_patch>\n"
        f"{candidate_patch.rstrip()}\n"
        "</candidate_patch>\n"
    )
    test_block = (
        "Scheduler-owned visible test evidence\n"
        "<visible_test_evidence>\n"
        f"{scheduled_test_evidence.rstrip()}\n"
        "</visible_test_evidence>\n"
    )
    focus_block = ""
    if focused_new_regression:
        focus_block = (
            "Focused repair mode: the scheduler proved at least one new_regression.\n"
            "- Only Read and Edit are available. Work only from the failing command/"
            "output, candidate diff, and files already named there.\n"
            "- Treat the first new_regression as the primary defect. Read only the "
            "smallest relevant region, then Edit the existing source immediately.\n"
            "- Do not redesign the solution or investigate unrelated APIs. Prefer "
            "reverting the offending hunk if a narrow repair is uncertain.\n"
        )
    api_instruction = (
        "- Do not search for other API usages in focused repair mode; the failing "
        "evidence and changed files are the complete scope.\n"
        if focused_new_regression
        else "- Search for an existing API usage before accepting unfamiliar calling syntax.\n"
    )
    return (
        f"{base_prompt.rstrip()}\n\n"
        "Current phase: verification and repair\n"
        f"- You have at most {verification_turns} turns. Do not restart broad exploration.\n"
        f"{patch_block}"
        f"{test_block}"
        f"{focus_block}"
        "- The patch above is authoritative even if plain `git diff` is empty because an "
        "earlier agent may have committed it. Do not inspect Git history.\n"
        "- Do not create another Git commit.\n"
        "- Bash is disabled in this phase. Do not attempt pytest, package installation, "
        "Git commands, or any host process; use Read and Edit to repair the files.\n"
        "- The scheduler compares every command on the unmodified baseline and patched "
        "candidate. Treat only `new_regression` (baseline passed, patched failed) as a "
        "regression introduced by this patch. `baseline_failure_persists` is pre-existing "
        "evidence, not a reason by itself to rewrite or abandon the patch.\n"
        "- A missing plan or runner error remains diagnostic information, not a reason to "
        "abandon the non-empty patch.\n"
        f"{api_instruction}"
        f"- Read at most {max_file_read_lines} source lines in one tool call.\n"
        f"- Keep each command output below about {max_tool_output_chars} characters; "
        "show only the first relevant failure and a short tail.\n"
        "- The parent scheduler regenerates and reruns repository-adapted commands after "
        "this phase. Do not create or replace a test-plan control file.\n"
        "- Finish after inspecting every changed source file relevant to the candidate. A "
        "generic response or empty patch is a failure.\n"
        "- Do not inspect or run hidden SWE-bench tests; use only repository-visible tests "
        "and reproductions derived from the problem statement.\n"
    )


def build_recovery_edit_gate_prompt(
    base_prompt: str,
    *,
    recovery_turns: int,
    implementation_handoff: str,
    source_context: str,
) -> str:
    """构造 Recovery 的强制 Edit 阶段，配合 CLI 仅开放 Edit 工具。

    此阶段位于任何新的探索之前。父进程提供 Implementation 已定位的公开结论和
    源码片段，模型必须直接尝试修改；Read/Grep/Glob/Bash/Write 均由 CLI 禁用，
    因而“尽早 Edit”不再只是可以被忽略的自然语言建议。
    """

    handoff = implementation_handoff.strip() or "No visible diagnosis was recorded."
    context = source_context.strip() or "No safe source excerpt was available."
    return (
        f"{base_prompt.rstrip()}\n\n"
        "Current phase: empty-patch recovery implementation — mandatory Edit gate\n"
        f"- You have at most {recovery_turns} turns. Your first tool call must be Edit.\n"
        "- Edit is the only filesystem tool available. Read, Grep, Glob, Bash, Write, "
        "and subagents are disabled by the parent process.\n"
        "- Use the handoff and exact source excerpts below to make the smallest plausible "
        "change to existing product source now. Do not respond with an explanation only.\n"
        "<implementation_handoff>\n"
        f"{handoff}\n"
        "</implementation_handoff>\n"
        "<source_context>\n"
        f"{context}\n"
        "</source_context>\n"
        "- Do not create a Git commit, test file, reproduction file, or control file.\n"
    )


def build_recovery_implementation_prompt(
    base_prompt: str,
    *,
    recovery_turns: int,
    max_file_read_lines: int,
    max_tool_output_chars: int,
    implementation_handoff: str = "",
) -> str:
    """为强制 Edit 未产出源码 patch 的情况构造受限 fallback 会话。

    Recovery 只复用原本预留给 Verification、但因没有候选补丁而无法使用的
    turns，因此不会扩大单题模型预算。上一阶段只交接可见结论和工具调用摘要，
    不传递 thinking 或大段工具输出。由于前置 gate 已真实尝试 Edit，本阶段只允许
    使用剩余预算做最小范围的补充读取，再交付源码修改。
    """

    handoff = implementation_handoff.strip() or (
        "No usable visible finding was produced by the previous session."
    )

    return (
        f"{base_prompt.rstrip()}\n\n"
        "Current phase: empty-patch recovery fallback\n"
        "Visible handoff from the previous implementation session\n"
        "<implementation_handoff>\n"
        f"{handoff}\n"
        "</implementation_handoff>\n"
        "- The mandatory Edit gate did not produce an existing-source patch. Do not "
        "repeat searches or file reads already summarized in the handoff.\n"
        f"- You have at most {recovery_turns} turns to make a minimal concrete source "
        "change that addresses the issue.\n"
        "- Bash is disabled in this phase. Use only targeted Read, Grep, and Glob for "
        "inspection; do not run tests or package installers.\n"
        "- The earlier gate already enforced an immediate Edit attempt. Use the remaining "
        "calls only to correct that attempt, then Edit existing product source.\n"
        "- Modify existing product source. A reproduction script, generated environment, "
        "test-only change, explanation, or empty working tree is not a fix.\n"
        "- Do not create a Git commit or `.agent-test-plan.json`.\n"
        f"- Read at most {max_file_read_lines} source lines in one tool call.\n"
        f"- Keep each command output below about {max_tool_output_chars} characters.\n"
        "- If evidence remains incomplete, still leave the safest minimal Edit supported "
        "by the issue and handoff before this phase ends.\n"
    )
