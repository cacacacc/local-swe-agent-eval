"""Safe, agent-facing representation of a SWE-bench task."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


class TaskValidationError(ValueError):
    """Raised when a dataset record cannot identify a reproducible task."""


_REPOSITORY_PATTERN = re.compile(
    r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$"
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{7,40}$")


@dataclass(frozen=True, slots=True)
class SWEbenchTask:
    """The minimum SWE-bench information exposed to a solving agent.

    Evaluation-only fields such as ``patch`` and ``test_patch`` are deliberately
    absent. Keeping this type small makes accidental solution leakage easier to
    detect in code review and tests.
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str

    def __post_init__(self) -> None:
        values = {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
        }
        for field_name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise TaskValidationError(
                    f"{field_name} must be a non-empty string"
                )

        if not _REPOSITORY_PATTERN.fullmatch(self.repo):
            raise TaskValidationError(
                "repo must have the GitHub 'owner/name' form"
            )
        if not _COMMIT_PATTERN.fullmatch(self.base_commit):
            raise TaskValidationError(
                "base_commit must be a 7-40 character hexadecimal Git commit"
            )

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SWEbenchTask":
        """Validate a raw dataset record and copy only approved fields."""

        required = (
            "instance_id",
            "repo",
            "base_commit",
            "problem_statement",
        )
        missing = [field for field in required if field not in record]
        if missing:
            raise TaskValidationError(
                f"missing required field(s): {', '.join(missing)}"
            )

        return cls(**{field: record[field] for field in required})

    def to_agent_payload(self) -> dict[str, str]:
        """Return the complete and only payload permitted in agent context."""

        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
        }

