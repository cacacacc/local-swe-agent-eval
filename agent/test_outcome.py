"""Normalize test output from different repositories into comparable failure signatures.

The Docker sandbox only executes reliably and preserves the output; this module then
identifies the failure summaries of pytest, Django unittest, and SymPy ``bin/test``
based on the actual argv. The scheduler compares signature sets rather than only exit
codes, so when the baseline already has unrelated failures, new failures introduced by
the candidate can still trigger focused repair.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from agent.test_sandbox import VisibleTestResult


_PYTEST_FAILURE = re.compile(
    r"^(?:FAILED|ERROR)\s+([^\s]+)",
    flags=re.MULTILINE,
)
_UNITTEST_FAILURE = re.compile(
    r"^(?:FAIL|ERROR):\s+(.+?)\s*$",
    flags=re.MULTILINE,
)
_SYMPY_FAILURE = re.compile(
    r"^_{3,}\s+([^\s]+\.py:[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]\n]+\])?)\s+_{3,}\s*$",
    flags=re.MULTILINE,
)
# unittest subTest arguments sometimes include a default object repr whose memory
# address differs across Python processes. Keep the object type and the other stable
# arguments and replace only the address, so baseline/candidate do not generate
# different signatures for the same test.
_VOLATILE_HEX_ADDRESS = re.compile(r"(?<=\bat )0x[0-9a-fA-F]+\b")


@dataclass(frozen=True, slots=True)
class TestOutcome:
    """The normalized status, failure test set, and parsing confidence of one test run."""

    status: str
    failure_signatures: frozenset[str] = frozenset()
    parser: str = "none"
    reliable: bool = False


def parse_test_outcome(
    argv: Sequence[str],
    result: VisibleTestResult,
) -> TestOutcome:
    """Parse the result according to the test entry point; return an explicit untrusted status when failures cannot be located reliably.

    A successful exit can be judged reliably without a failure summary. Timeout is
    likewise a definite state but has no test ID that could be set-subtracted against
    another failure; infrastructure errors and commands that never started do not
    constitute test evidence at all.
    """

    parser, pattern = _select_parser(argv)
    if not result.evidence_valid:
        return TestOutcome(status="unavailable", parser=parser, reliable=False)
    if result.timed_out:
        return TestOutcome(status="timeout", parser=parser, reliable=True)
    if result.exit_code == 0:
        return TestOutcome(status="passed", parser=parser, reliable=True)

    signatures = frozenset(
        normalized
        for match in pattern.findall(result.output)
        if (normalized := _normalize_signature(match))
    )
    return TestOutcome(
        status="failed",
        failure_signatures=signatures,
        parser=parser,
        # A non-zero exit without any test ID may be a collection error, a process
        # crash, or an uncovered output format. In that case we cannot claim the
        # candidate added no regression based on the same exit code alone.
        reliable=bool(signatures),
    )


def _select_parser(argv: Sequence[str]) -> tuple[str, re.Pattern[str]]:
    """Select a stable parser from the parent-generated argv, falling back to pytest style for unknown commands."""

    normalized = tuple(argv)
    if len(normalized) >= 2 and normalized[1] == "tests/runtests.py":
        return "django-unittest", _UNITTEST_FAILURE
    if len(normalized) >= 2 and normalized[1] == "bin/test":
        return "sympy-bin-test", _SYMPY_FAILURE
    if len(normalized) >= 3 and normalized[1:3] == ("-m", "pytest"):
        return "pytest", _PYTEST_FAILURE
    return "generic-pytest-summary", _PYTEST_FAILURE


def _normalize_signature(value: str) -> str:
    """Remove process-related addresses and collapse whitespace into a stable cross-container failing test ID."""

    without_addresses = _VOLATILE_HEX_ADDRESS.sub("0xADDR", value)
    return " ".join(without_addresses.strip().split())
