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
        "- After making the source fix, if the issue describes a concrete edge case, "
        "use Write to submit `.agent-issue-repro.py`: at most 30 lines of Python, "
        "with one assert of the expected public behavior. The parent removes this "
        "file from the patch and compares it on baseline and candidate in offline "
        "Docker. Do not run it on the host or use hidden tests. Do not delay the source "
        "fix solely to write a reproduction.\n"
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
    focused_syntax_error: bool = False,
    focused_issue_reproduction: bool = False,
    syntax_preflight_evidence: str = "",
    issue_reproduction_evidence: str = "",
    handoff_text: str = "",
) -> str:
    """Inject current patch, test evidence, and cumulative stage state into the repair prompt."""
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
    handoff_block = (
        "Parent-maintained stage handoff\n"
        "<phase_handoff>\n"
        f"{handoff_text.rstrip()}\n"
        "</phase_handoff>\n"
        if handoff_text.strip()
        else ""
    )
    focus_block = ""
    if focused_syntax_error:
        focus_block = (
            "Focused syntax repair: the parent Docker preflight proved a new Python "
            "syntax error relative to the baseline. Only Read and Edit are available. "
            "Read the reported source location, then correct that syntax before other work.\n"
        )
    elif focused_new_regression:
        focus_block = (
            "Focused repair mode: the scheduler proved at least one new_regression.\n"
            "- Only Read and Edit are available. Work only from the failing command/"
            "output, candidate diff, and files already named there.\n"
            "- Treat the first new_regression as the primary defect. Read only the "
            "smallest relevant region, then Edit the existing source immediately.\n"
            "- Do not redesign the solution or investigate unrelated APIs. Prefer "
            "reverting the offending hunk if a narrow repair is uncertain.\n"
        )
    elif focused_issue_reproduction:
        focus_block = (
            "Focused issue reproduction repair: the parent-run boundary assertion "
            "still fails on the current candidate. Only Read and Edit are available. "
            "Inspect the asserted input/output and the changed source, then make the "
            "smallest correction; do not treat this one assertion as full proof.\n"
        )
    api_instruction = (
        "- Do not search for other API usages in focused repair mode; the failing "
        "evidence and changed files are the complete scope.\n"
        if focused_new_regression or focused_syntax_error or focused_issue_reproduction
        else "- Search for an existing API usage before accepting unfamiliar calling syntax.\n"
    )
    syntax_block = (
        "Docker Python syntax preflight\n<syntax_preflight>\n"
        f"{syntax_preflight_evidence.rstrip()}\n</syntax_preflight>\n"
        if syntax_preflight_evidence else ""
    )
    reproduction_block = (
        "Parent-run issue boundary reproduction\n<issue_reproduction>\n"
        f"{issue_reproduction_evidence.rstrip()}\n</issue_reproduction>\n"
        if issue_reproduction_evidence else ""
    )
    return (
        f"{base_prompt.rstrip()}\n\n"
        "Current phase: verification and repair\n"
        f"- You have at most {verification_turns} turns. Do not restart broad exploration.\n"
        f"{patch_block}"
        f"{test_block}"
        f"{syntax_block}"
        f"{reproduction_block}"
        f"{handoff_block}"
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


def build_recovery_read_edit_gate_prompt(
    base_prompt: str,
    *,
    recovery_turns: int,
    implementation_handoff: str,
    source_context: str,
) -> str:
    """Build a multi-file Recovery gate restricted to ordered source Reads and Edits.

    The model may inspect and modify as many existing product-source files as the fix
    requires. The parent process validates the complete tool trajectory, including the
    invariant that each edited file was read earlier in the same gate.
    """

    handoff = implementation_handoff.strip() or "No visible diagnosis was recorded."
    context = source_context.strip() or "No safe source excerpt was available."
    return (
        f"{base_prompt.rstrip()}\n\n"
        "Current phase: empty-patch recovery — Multi-file Read/Edit Gate\n"
        f"- You have at most {recovery_turns} turns. Only Read and Edit are available. "
        "Use them to leave the smallest plausible product-source fix.\n"
        "- You may read and edit multiple existing product-source files when the issue "
        "requires it. Before every Edit, Read that same file earlier in this gate.\n"
        "- Do not read or edit tests, reproductions, generated artifacts, documentation, "
        "or control files. Do not finish with an explanation instead of a source Edit.\n"
        "- The parent process validates every tool event and rejects unsupported tools, "
        "invalid paths, or an Edit without a preceding same-file Read.\n"
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
    last_chance: bool = False,
) -> str:
    """Build a restricted fallback or last-chance Recovery session.

    Recovery only reuses the turns originally reserved for Verification but unusable
    because no candidate patch exists, so it does not enlarge the per-task model
    budget. The previous stage hands off only visible findings and a tool-call summary,
    never thinking or large tool outputs.
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
        "- The Multi-file Read/Edit Gate did not produce a validated existing-"
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
