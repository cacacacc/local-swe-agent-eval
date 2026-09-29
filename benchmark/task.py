"""Define the safe SWE-bench task object exposed to the solving Agent.

Raw SWE-bench records may also contain evaluation-only fields such as reference
patches and hidden test patches. This module keeps only the four fields needed for
solving via an explicit whitelist, reducing the risk of answer leakage at the data
structure level.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


class TaskValidationError(ValueError):
    """Raised when a data record cannot uniquely and reproducibly describe a task."""


# The repository must use GitHub's ``owner/name`` form to avoid arbitrary URL or path injection.
_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
# Git supports short SHAs; 7 to 40 characters cover common short SHAs and full SHA-1.
_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{7,40}$")


@dataclass(frozen=True, slots=True)
class SWEbenchTask:
    """The minimal task representation allowed into the solving Agent.

    Evaluation fields such as ``patch`` and ``test_patch`` are deliberately excluded.
    Keeping the data type small makes it easier for code review and automated tests to
    catch accidental data leakage.
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str

    def __post_init__(self) -> None:
        """Validate all key identifiers immediately after the immutable dataclass is created."""

        # Check types and empty strings uniformly first, so later regex matching does not produce ambiguous errors.
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
        """Validate the raw dataset record and copy only the whitelisted fields.

        Even if ``record`` contains a reference patch, this method never stores it in
        the object.
        """

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

        # Explicitly projecting onto the required fields is the anti-leakage boundary; do not change it to ``cls(**record)``.
        return cls(**{field: record[field] for field in required})

    def to_agent_payload(self) -> dict[str, str]:
        """Return the complete, unambiguous data payload allowed into the Agent context."""

        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
        }
