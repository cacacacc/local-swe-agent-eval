"""Deterministic Mock Agent used to validate the experiment pipeline.

While Phase 2 has not yet connected a real LLM, this implementation verifies that
file modification, event recording, patch collection, and failure handling all
work correctly. It does not simulate model capability and must not be treated as a
formal SWE-bench experiment result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.task import SWEbenchTask


class MockAgentError(RuntimeError):
    """Raised when the Mock Agent cannot safely produce the predetermined change."""


def _utc_now() -> str:
    """Return an ISO 8601 timestamp with UTC offset so logs compare across machines."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class ObservableEvent:
    """One externally observable behavior, excluding hidden model reasoning or chain-of-thought."""

    sequence: int
    timestamp: str
    event_type: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert into a plain dict that can be written directly to trajectory JSON."""

        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class MockAgentResult:
    """Mock execution result whose overall shape matches the future real Agent runner."""

    exit_code: int
    agent_log: str
    test_output: str
    events: tuple[ObservableEvent, ...]


class MockAgentRunner:
    """Create a predictably named untracked file in a prepared repository."""

    def __init__(self, output_name: str = "mock_agent_change.txt") -> None:
        """Validate that the output is a single relative file name at the repository root."""

        output_path = Path(output_name)
        if output_path.is_absolute() or len(output_path.parts) != 1:
            raise ValueError("output_name must be a single relative file name")
        if output_name in {"", ".", ".."}:
            raise ValueError("output_name must identify a file")
        self.output_name = output_name

    def run(self, task: SWEbenchTask, repository: Path | str) -> MockAgentResult:
        """Perform one deterministic change and return the log, pseudo test output, and event sequence."""

        repository_path = Path(repository).resolve()
        if not repository_path.is_dir():
            raise MockAgentError(f"repository does not exist: {repository_path}")

        output_path = repository_path / self.output_name
        # Never overwrite an existing file: this protects user data and also makes
        # repeat-run problems visible immediately.
        if output_path.exists():
            raise MockAgentError(
                f"mock output already exists; refusing to overwrite: {output_path}"
            )

        content = (
            "This file was created by the deterministic Phase 2 mock agent.\n"
            f"instance_id={task.instance_id}\n"
            f"base_commit={task.base_commit}\n"
        )
        output_path.write_text(content, encoding="utf-8")

        # The trajectory records only verifiable behaviors such as write file / run
        # tests / exit.
        events = (
            ObservableEvent(
                sequence=1,
                timestamp=_utc_now(),
                event_type="file_write",
                details={
                    "path": self.output_name,
                    "bytes_written": len(content.encode("utf-8")),
                },
            ),
            ObservableEvent(
                sequence=2,
                timestamp=_utc_now(),
                event_type="test_run",
                details={"command": "mock-test", "exit_code": 0},
            ),
            ObservableEvent(
                sequence=3,
                timestamp=_utc_now(),
                event_type="agent_exit",
                details={"exit_code": 0},
            ),
        )
        return MockAgentResult(
            exit_code=0,
            agent_log=(
                f"Mock agent inspected task {task.instance_id}.\n"
                f"Created {self.output_name}.\n"
                "Mock agent finished with exit code 0.\n"
            ),
            test_output="mock-test: passed\n",
            events=events,
        )
