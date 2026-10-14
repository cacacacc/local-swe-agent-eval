"""Consume a bounded issue-derived reproduction without trusting model test commands."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import shutil


REPRODUCTION_FILENAME = ".agent-issue-repro.py"
REPRODUCTION_PLACEHOLDER = "# Replace this line with a short issue assertion.\n"
_MAX_REPRODUCTION_BYTES = 8192
_MAX_REPRODUCTION_LINES = 60


@dataclass(frozen=True, slots=True)
class IssueReproductionRequest:
    """Represent a one-shot script request separately from its Docker execution result."""

    status: str
    code: str = ""
    error: str | None = None

    @property
    def accepted(self) -> bool:
        """Only a short, parseable script with an assertion may be scheduled."""

        return self.status == "accepted"


def consume_issue_reproduction(repository: Path | str) -> IssueReproductionRequest:
    """Read and remove the fixed-name model artifact before any patch is collected.

    The parent never accepts an executable path or test argv from this file. Its
    contents can run only as ``python -c`` inside the offline SWE-bench container.
    A malformed file is still removed so it cannot contaminate predictions.
    """

    path = Path(repository).resolve() / REPRODUCTION_FILENAME
    if not path.exists() and not path.is_symlink():
        return IssueReproductionRequest("missing")
    try:
        if path.is_symlink() or not path.is_file():
            return IssueReproductionRequest("rejected", error="reproduction must be a regular file")
        if path.stat().st_size > _MAX_REPRODUCTION_BYTES:
            return IssueReproductionRequest("rejected", error="reproduction exceeds 8192 bytes")
        try:
            code = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            return IssueReproductionRequest("rejected", error=f"cannot read reproduction: {error}")
        if not code.strip() or code == REPRODUCTION_PLACEHOLDER:
            # Implementation receives a parent-owned Edit target; leaving the
            # placeholder untouched is missing, not an invalid submission.
            return IssueReproductionRequest("missing")
        if len(code.splitlines()) > _MAX_REPRODUCTION_LINES:
            return IssueReproductionRequest("rejected", error="reproduction exceeds 60 lines")
        try:
            tree = ast.parse(code, filename=REPRODUCTION_FILENAME)
        except SyntaxError as error:
            return IssueReproductionRequest("rejected", error=f"invalid Python syntax: {error.msg}")
        if not any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
            return IssueReproductionRequest("rejected", error="reproduction requires an assert")
        return IssueReproductionRequest("accepted", code=code)
    finally:
        # This precise control path is owned by the isolated agent workspace. Never
        # follow a symlink or remove any neighboring repository content.
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
