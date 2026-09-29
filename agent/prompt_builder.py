"""Generate an auditable Prompt for each SWE-bench task from a fixed template.

The template is validated once at construction time, and rendering accepts only the
safe fields of ``SWEbenchTask``, ensuring that only the problem data varies between
tasks while the experiment instructions stay identical.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from benchmark.task import SWEbenchTask


class PromptTemplateError(ValueError):
    """Raised when the Prompt template is empty, unreadable, or missing required placeholders."""


class PromptBuilder:
    """Load and validate the template, then render using only approved task fields."""

    REQUIRED_PLACEHOLDERS = (
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
    )

    def __init__(self, template_text: str) -> None:
        """Normalize newlines and confirm that all four task fields have explicit placeholders."""

        # Normalize to LF so Windows/WSL line-ending differences do not change the prompt hash.
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
        """Read the template from a UTF-8 text file, converting filesystem errors into domain errors."""

        source = Path(path)
        try:
            return cls(source.read_text(encoding="utf-8"))
        except OSError as error:
            raise PromptTemplateError(f"cannot read prompt template {source}: {error}") from error

    def build(self, task: SWEbenchTask) -> str:
        """Render a single task and guarantee the result ends with exactly one newline."""

        # ``to_agent_payload`` is the only data entry point, so evaluation-only fields
        # cannot be substituted into the template.
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
    """Append the patch delivery point and context budget guardrails for the implementation phase.

    Claude Code does not yet offer a reliable per-call hard limit on Read/Bash output,
    so these limits are written as an auditable phase protocol instead. Real tests are
    always scheduled by the parent process after the candidate patch is produced;
    Implementation no longer carries the up-front testing that is easily misled by the
    host environment.
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
    """Inject the candidate patch and test evidence, building the repair Prompt based on whether a new regression occurred."""
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
        "candidate. `new_regression` means either baseline passed while candidate failed, "
        "or the candidate added a parsed failing test ID on top of existing baseline "
        "failures. `baseline_failure_persists` contains no newly observed failing ID and "
        "is not a reason by itself to rewrite or abandon the patch.\n"
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


def build_recovery_read_gate_prompt(
    base_prompt: str,
    *,
    implementation_handoff: str,
    source_context: str,
) -> str:
    """Build the first step of the Recovery state machine, requiring and allowing exactly one target-source Read.

    The parent process subsequently validates the actual tool event; only a read of
    existing in-repository source proceeds to the Edit-only second step in the same
    session. The Prompt therefore only provides the model context, while the parent
    process enforces the sequencing.
    """

    handoff = implementation_handoff.strip() or "No visible diagnosis was recorded."
    context = source_context.strip() or "No safe source excerpt was available."
    return (
        f"{base_prompt.rstrip()}\n\n"
        "Current phase: empty-patch recovery — mandatory Read step\n"
        "- This invocation has exactly one action: call Read exactly once on the most "
        "relevant existing product-source file named below.\n"
        "- Read is the only available tool. Do not answer with text, request another "
        "tool, or read a test, reproduction, generated, or control file.\n"
        "- The parent process validates the tool event and will resume this exact session "
        "with only Edit available if the path is valid.\n"
        "<implementation_handoff>\n"
        f"{handoff}\n"
        "</implementation_handoff>\n"
        "<source_context>\n"
        f"{context}\n"
        "</source_context>\n"
        "- Do not create a Git commit, test file, reproduction file, or control file.\n"
    )


def build_recovery_edit_gate_prompt(
    base_prompt: str,
    *,
    target_file: str,
) -> str:
    """Build the second step of the Recovery state machine, allowing only an Edit of the same file read in the first step."""

    return (
        f"{base_prompt.rstrip()}\n\n"
        "Current phase: empty-patch recovery — mandatory Edit step\n"
        f"- The validated Read step selected `{target_file}`.\n"
        "- This invocation has exactly one action: call Edit exactly once on that same "
        "file and leave the smallest plausible product-source fix.\n"
        "- Edit is the only available tool. Do not answer with text, request Read, or "
        "modify a different path.\n"
        "- Use the issue, handoff, and source content already present in this resumed "
        "session. If evidence is incomplete, still make the safest minimal Edit.\n"
        "- Do not create a Git commit, test file, reproduction file, or control file.\n"
    )


def build_recovery_implementation_prompt(
    base_prompt: str,
    *,
    recovery_turns: int,
    max_file_read_lines: int,
    max_tool_output_chars: int,
    implementation_handoff: str = "",
    last_chance: bool = False,
) -> str:
    """Build a restricted fallback session for the case where the mandatory Edit produced no source patch.

    Recovery only reuses the turns originally reserved for Verification but unusable
    because no candidate patch exists, so it does not enlarge the per-task model
    budget. The previous stage hands off only visible findings and a tool-call summary,
    never thinking or large tool outputs. Because the upstream gate already made a real
    Edit attempt, this stage only allows minimal supplementary reads with the remaining
    budget before delivering a source change.
    """

    handoff = implementation_handoff.strip() or (
        "No usable visible finding was produced by the previous session."
    )

    phase_name = (
        "empty-patch recovery last chance"
        if last_chance
        else "empty-patch recovery fallback"
    )
    budget_note = (
        "- Earlier Recovery stages still produced no source patch, so the turns reserved "
        "for regression repair are being reused as a final implementation attempt.\n"
        if last_chance
        else ""
    )

    return (
        f"{base_prompt.rstrip()}\n\n"
        f"Current phase: {phase_name}\n"
        "Visible handoff from the previous implementation session\n"
        "<implementation_handoff>\n"
        f"{handoff}\n"
        "</implementation_handoff>\n"
        "- The mandatory Read-Edit state machine did not produce a validated existing-"
        "source patch. Do not "
        "repeat searches or file reads already summarized in the handoff.\n"
        f"{budget_note}"
        f"- You have at most {recovery_turns} turns to make a minimal concrete source "
        "change that addresses the issue.\n"
        "- Bash is disabled in this phase. Use only targeted Read, Grep, and Glob for "
        "inspection; do not run tests or package installers.\n"
        "- The earlier gate either failed sequence validation or produced no usable patch. "
        "Use the remaining calls to make and correct one concrete product-source Edit.\n"
        "- Modify existing product source. A reproduction script, generated environment, "
        "test-only change, explanation, or empty working tree is not a fix.\n"
        "- Do not create a Git commit or `.agent-test-plan.json`.\n"
        f"- Read at most {max_file_read_lines} source lines in one tool call.\n"
        f"- Keep each command output below about {max_tool_output_chars} characters.\n"
        "- If evidence remains incomplete, still leave the safest minimal Edit supported "
        "by the issue and handoff before this phase ends.\n"
    )
