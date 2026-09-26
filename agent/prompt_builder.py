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
    baseline_test_evidence: str,
) -> str:
    """把基线 Docker 测试证据注入实现阶段，并追加补丁交付护栏。

    Claude Code 暂不提供可靠的逐次 Read/Bash 输出硬上限，因此这里把限制写成
    可审计的阶段协议；运行器先独立规划并执行基线测试，再把真实输出交给实现
    会话，避免模型在没有复现证据时直接猜测修复。
    """

    return (
        f"{base_prompt.rstrip()}\n\n"
        "Current phase: implementation\n"
        f"- You have at most {implementation_turns} turns in this phase.\n"
        "- The parent scheduler already ran the target and adjacent-regression tests "
        "against the unmodified baseline. Diagnose the issue by combining the issue "
        "statement with this real evidence before editing code.\n"
        "<baseline_visible_test_evidence>\n"
        f"{baseline_test_evidence.rstrip()}\n"
        "</baseline_visible_test_evidence>\n"
        "- Produce a non-empty candidate patch before this phase ends.\n"
        "- Do not create a Git commit; leave the candidate change in the working tree.\n"
        "- Start with rg or another targeted search; do not dump whole large files.\n"
        f"- Read at most {max_file_read_lines} source lines in one tool call.\n"
        f"- Keep each command output below about {max_tool_output_chars} characters.\n"
        "- Before assuming how an internal API works, find an existing repository usage.\n"
        "- Do not run pytest, tox, project test scripts, package installers, or the "
        "visible-test helper from Bash. Host-side test attempts are invalid evidence; "
        "the parent scheduler owns all test execution.\n"
        "- Do not create or replace `.agent-test-plan.json`; the parent scheduler will "
        "rerun the already accepted baseline plan after your changes.\n"
        f"- A fresh verification session owns the final {verification_turns} turns, so "
        "leave the working tree with your best concrete patch even if local dependencies "
        "prevent tests from running.\n"
    )


def build_test_planning_phase_prompt(
    base_prompt: str,
    *,
    planning_turns: int,
    max_file_read_lines: int,
    previous_plan_error: str,
    test_inventory: str,
) -> str:
    """注入父进程生成的测试索引，构造实现前的短规划会话 Prompt。"""

    return (
        f"{base_prompt.rstrip()}\n\n"
        "Current phase: pre-implementation visible-test planning\n"
        f"- You have at most {planning_turns} turns. Do not modify product or test source.\n"
        "- Bash is disabled; Glob and Grep are unavailable. Use the parent-generated tracked-test "
        "inventory below, then Read only the most relevant listed files.\n"
        "- Your only deliverable is `.agent-test-plan.json`; create it with Write even "
        "if it does not exist.\n"
        "- The JSON object must contain exactly `target_argv` and `regression_argv`. "
        "Each value is a non-empty argv string array, not a shell command.\n"
        "- Example: `"
        "{"
        "\"target_argv\":[\"python\",\"-m\",\"pytest\",\"tests/test_one.py::test_bug\"],"
        "\"regression_argv\":[\"python\",\"-m\",\"pytest\",\"tests/test_one.py\"]}`.\n"
        "- `target_argv` must run the smallest repository-visible reproduction or focused "
        "test relevant to the reported bug. On the unmodified baseline it must fail with "
        "exit code 1. Do not catch or print an exception while returning exit code 0.\n"
        "- `regression_argv` must be a different, broader command that runs the nearest "
        "existing test module or suite, so adjacent behavior is checked too; it must pass "
        "with exit code 0 on the unmodified baseline.\n"
        "- Do not use Git, Docker, package installers, network tools, pipes, redirects, "
        "or shell operators.\n"
        f"- Read at most {max_file_read_lines} lines per tool call.\n"
        f"Previous submission status: {previous_plan_error}\n"
        "<tracked_test_inventory>\n"
        f"{test_inventory.rstrip()}\n"
        "</tracked_test_inventory>\n"
        "No product patch exists yet. Select commands using the issue statement and "
        "the unmodified repository, so their baseline output can guide implementation.\n"
        "Finish immediately after writing the valid control file.\n"
    )


def build_verification_phase_prompt(
    base_prompt: str,
    *,
    verification_turns: int,
    max_file_read_lines: int,
    max_tool_output_chars: int,
    candidate_patch: str = "",
    scheduled_test_evidence: str = "",
) -> str:
    """注入候选补丁和调度测试证据，构造无 Bash 的修复会话 Prompt。"""
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
    return (
        f"{base_prompt.rstrip()}\n\n"
        "Current phase: verification and repair\n"
        f"- You have at most {verification_turns} turns. Do not restart broad exploration.\n"
        f"{patch_block}"
        f"{test_block}"
        "- The patch above is authoritative even if plain `git diff` is empty because an "
        "earlier agent may have committed it. Do not inspect Git history.\n"
        "- Do not create another Git commit.\n"
        "- Bash is disabled in this phase. Do not attempt pytest, package installation, "
        "Git commands, or any host process; use Read and Edit to repair the files.\n"
        "- Treat the actual traceback or assertion as authoritative and repair the patch.\n"
        "- Search for an existing API usage before accepting unfamiliar calling syntax.\n"
        f"- Read at most {max_file_read_lines} source lines in one tool call.\n"
        f"- Keep each command output below about {max_tool_output_chars} characters; "
        "show only the first relevant failure and a short tail.\n"
        "- The parent scheduler automatically reruns both accepted argv commands. Only "
        "if a test target must change, use Write to create `.agent-test-plan.json` with "
        "new `target_argv` and `regression_argv` arrays. Never use a shell command string.\n"
        "- The run is locally completed only if both final Docker commands exit 0.\n"
        "- Finish after inspecting every changed source file relevant to the candidate. A "
        "generic response or empty patch is a failure.\n"
        "- Do not inspect or run hidden SWE-bench tests; use only repository-visible tests "
        "and reproductions derived from the problem statement.\n"
    )
