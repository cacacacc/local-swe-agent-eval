"""A deterministic agent substitute for testing the experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.task import SWEbenchTask


class MockAgentError(RuntimeError):
    """Raised when a mock run cannot safely create its deterministic change."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class ObservableEvent:
    """One externally observable action, never hidden model reasoning."""

    sequence: int
    timestamp: str
    event_type: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class MockAgentResult:
    """Outputs with the same broad shape expected from a real agent runner."""

    exit_code: int
    agent_log: str
    test_output: str
    events: tuple[ObservableEvent, ...]


class MockAgentRunner:
    """Create one predictable untracked file in a prepared task repository."""

    def __init__(self, output_name: str = "mock_agent_change.txt") -> None:
        output_path = Path(output_name)
        if output_path.is_absolute() or len(output_path.parts) != 1:
            raise ValueError("output_name must be a single relative file name")
        if output_name in {"", ".", ".."}:
            raise ValueError("output_name must identify a file")
        self.output_name = output_name

    def run(self, task: SWEbenchTask, repository: Path | str) -> MockAgentResult:
        repository_path = Path(repository).resolve()
        if not repository_path.is_dir():
            raise MockAgentError(f"repository does not exist: {repository_path}")

        output_path = repository_path / self.output_name
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

