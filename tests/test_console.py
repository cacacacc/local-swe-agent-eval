"""Verify the stable text structure of terminal visualization in a plain log stream."""

from io import StringIO

from tracking.console import ConsoleReporter, format_duration


def test_format_duration_uses_fixed_width_clock() -> None:
    """Long-running experiment durations must use fixed width so summary tables do not jump with the values."""

    assert format_duration(0) == "00:00:00"
    assert format_duration(3661.9) == "01:01:01"


def test_reporter_uses_thirty_second_heartbeat_by_default() -> None:
    """All long-task entry points should share the 30-second heartbeat so output cadence does not drift across scripts."""

    reporter = ConsoleReporter(StringIO())

    assert reporter.heartbeat_seconds == 30.0


def test_reporter_prints_stage_task_and_aligned_table() -> None:
    """Non-TTY output should still include stage/task progress and a readable summary table."""

    stream = StringIO()
    reporter = ConsoleReporter(stream)
    reporter.banner("Evaluation", "batch-001")
    reporter.stage(2, 4, "Build predictions")
    reporter.task(3, 10, "owner__repo-3")
    reporter.table(("#", "status"), ((1, "resolved"), (2, "unresolved")))

    output = stream.getvalue()
    assert "Evaluation" in output
    assert "Stage 2/4" in output
    assert "Task 3/10" in output
    assert "resolved" in output and "unresolved" in output
