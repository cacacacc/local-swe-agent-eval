"""Terminal stage panel, progress bars, heartbeats, and summary tables with no third-party dependencies."""

from __future__ import annotations

from contextlib import contextmanager
import sys
from threading import Event, Thread
import time
from typing import Iterator, Sequence, TextIO


def format_duration(seconds: float) -> str:
    """Format wall-clock seconds as ``HH:MM:SS`` for quick terminal scanning."""

    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ConsoleReporter:
    """Show long-running experiment progress as stable text; non-interactive output produces no periodic noise."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        heartbeat_seconds: float = 30.0,
    ) -> None:
        """Store the output stream; long operations report status every 30 seconds on a real TTY."""

        self.stream = sys.stdout if stream is None else stream
        self.heartbeat_seconds = heartbeat_seconds

    def line(self, text: str = "") -> None:
        """Print a line immediately so buffering does not make long tasks look stuck."""

        print(text, file=self.stream, flush=True)

    def banner(self, title: str, subtitle: str | None = None) -> None:
        """Print a pipeline banner identifying the current run or batch."""

        width = max(60, len(title) + 4, len(subtitle or "") + 4)
        self.line("=" * width)
        self.line(f"  {title}")
        if subtitle:
            self.line(f"  {subtitle}")
        self.line("=" * width)

    def stage(self, current: int, total: int, label: str) -> None:
        """Print a stage progress bar; the fixed stage count separates Agent and Docker time."""

        completed = max(0, min(current - 1, total))
        bar = "█" * completed + "░" * (total - completed)
        self.line(f"\n[{bar}] Stage {current}/{total}  {label}")

    def task(self, current: int, total: int, instance_id: str) -> None:
        """Print a per-task progress bar for batch solving plus the current instance ID."""

        width = 20
        filled = int(width * (current - 1) / total)
        bar = "█" * filled + "░" * (width - filled)
        self.line(f"\n[{bar}] Task {current}/{total}  {instance_id}")

    @contextmanager
    def activity(self, label: str) -> Iterator[None]:
        """Wrap a long operation and show its elapsed time; on a TTY print a periodic "still running" heartbeat."""

        started = time.monotonic()
        stopped = Event()
        self.line(f"  ▶ {label}")

        def heartbeat() -> None:
            # Event.wait doubles as an interruptible sleep, so we never wait a full heartbeat cycle at exit.
            while not stopped.wait(self.heartbeat_seconds):
                elapsed = format_duration(time.monotonic() - started)
                self.line(f"  … {label}, running ({elapsed})")

        interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        worker = Thread(target=heartbeat, daemon=True) if interactive else None
        if worker is not None:
            worker.start()
        try:
            yield
        except Exception:
            elapsed = format_duration(time.monotonic() - started)
            self.line(f"  ✗ {label} failed ({elapsed})")
            raise
        else:
            elapsed = format_duration(time.monotonic() - started)
            self.line(f"  ✓ {label} completed ({elapsed})")
        finally:
            stopped.set()
            if worker is not None:
                worker.join(timeout=1)

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
        """Print an aligned plain-text table; readable in WSL, logs, and CI without Rich."""

        rendered = [[str(cell) for cell in row] for row in rows]
        widths = [len(header) for header in headers]
        for row in rendered:
            if len(row) != len(headers):
                raise ValueError("table row width does not match headers")
            widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

        def format_row(row: Sequence[str]) -> str:
            return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

        self.line(format_row(list(headers)))
        self.line("-+-".join("-" * width for width in widths))
        for row in rendered:
            self.line(format_row(row))
