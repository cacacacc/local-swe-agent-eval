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
    baseline_test_command: str,
) -> str:
    """为实现阶段追加一次基线测试、补丁交付点和上下文预算护栏。

    Claude Code 暂不提供可靠的逐次 Read/Bash 输出硬上限，因此这里把限制写成
    可审计的阶段协议。Implementation 在修改源码前通过固定 helper 运行一次无网络
    Docker 测试；之后完全沿用 8/20 版本的补交计划、测试和 verification 流程。
    """

    return (
        f"{base_prompt.rstrip()}\n\n"
        "Current phase: implementation\n"
        f"- You have at most {implementation_turns} turns in this phase.\n"
        "- Before using Edit or Write on product or test source, run exactly one focused "
        "baseline test with the Docker helper below. You may inspect a small number of "
        "files first only to identify the correct visible test target. The command is "
        "task-specific: instance, commit, Docker image, timeout, and repository are already "
        "bound. Copy it exactly and append a complete test command argv.\n"
        f"  Pytest example: `{baseline_test_command} python -m pytest "
        "tests/test_module.py::test_case`\n"
        f"  Repository-runner example: `{baseline_test_command} "
        "./tests/runtests.py app.tests`\n"
        "  Do not prepend `python` before the helper itself, and do not pass a bare `.py` "
        "test path without its test runner.\n"
        "- Choose the smallest reproduction or focused repository test relevant to the "
        "issue. Treat its real output together with the issue statement as evidence, then "
        "diagnose and modify the code. Do not rerun the helper in this phase.\n"
        "- Produce a non-empty candidate patch before this phase ends.\n"
        "- Do not create a Git commit; leave the candidate change in the working tree.\n"
        "- Start with rg or another targeted search; do not dump whole large files.\n"
        f"- Read at most {max_file_read_lines} source lines in one tool call.\n"
        f"- Keep each command output below about {max_tool_output_chars} characters.\n"
        "- Before assuming how an internal API works, find an existing repository usage.\n"
        "- Do not run pytest, tox, project test scripts, or package installers directly "
        "on the host. The single helper call above is the only test command allowed in "
        "this phase; the parent scheduler owns all later test execution.\n"
        "- Before finishing, use Write to create `.agent-test-plan.json` with exactly "
        "`target_argv` and `regression_argv`. The first command targets the bug or a "
        "focused reproduction; the second runs the nearest existing test module or "
        "suite for regression coverage. Both values are argv arrays, never shell command "
        "strings. Example: `"
        "{"
        "\"target_argv\":[\"python\",\"-m\",\"pytest\",\"tests/test_one.py::test_bug\"],"
        "\"regression_argv\":[\"python\",\"-m\",\"pytest\",\"tests/test_one.py\"]}`. "
        "The control file is consumed and excluded from the patch.\n"
        f"- A fresh verification session owns the final {verification_turns} turns, so "
        "leave the working tree with your best concrete patch even if local dependencies "
        "prevent tests from running.\n"
    )


def build_test_planning_phase_prompt(
    base_prompt: str,
    *,
    planning_turns: int,
    max_file_read_lines: int,
    candidate_patch: str,
    previous_plan_error: str,
) -> str:
    """构造只负责交付两条测试 argv 的短会话 Prompt。"""

    return (
        f"{base_prompt.rstrip()}\n\n"
        "Current phase: mandatory visible-test planning\n"
        f"- You have at most {planning_turns} turns. Do not modify product or test source.\n"
        "- Bash is disabled. Use Read, Glob, and Grep only to locate tests.\n"
        "- Your only deliverable is `.agent-test-plan.json`; create it with Write even "
        "if it does not exist.\n"
        "- The JSON object must contain exactly `target_argv` and `regression_argv`. "
        "Each value is a non-empty argv string array, not a shell command.\n"
        "- Example: `"
        "{"
        "\"target_argv\":[\"python\",\"-m\",\"pytest\",\"tests/test_one.py::test_bug\"],"
        "\"regression_argv\":[\"python\",\"-m\",\"pytest\",\"tests/test_one.py\"]}`.\n"
        "- `target_argv` must run the smallest repository-visible reproduction or focused "
        "test relevant to the reported bug.\n"
        "- `regression_argv` must be a different, broader command that runs the nearest "
        "existing test module or suite, so adjacent behavior is checked too.\n"
        "- Do not use Git, Docker, package installers, network tools, pipes, redirects, "
        "or shell operators.\n"
        f"- Read at most {max_file_read_lines} lines per tool call.\n"
        f"Previous submission status: {previous_plan_error}\n"
        "Candidate patch:\n"
        "<candidate_patch>\n"
        f"{candidate_patch.rstrip()}\n"
        "</candidate_patch>\n"
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
        "- Finish after inspecting every changed source file relevant to the candidate. A "
        "generic response or empty patch is a failure.\n"
        "- Do not inspect or run hidden SWE-bench tests; use only repository-visible tests "
        "and reproductions derived from the problem statement.\n"
    )
