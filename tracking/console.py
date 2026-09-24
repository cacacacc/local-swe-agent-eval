"""提供无第三方依赖的终端阶段面板、进度条、心跳和汇总表。"""

from __future__ import annotations

from contextlib import contextmanager
import sys
from threading import Event, Thread
import time
from typing import Iterator, Sequence, TextIO


def format_duration(seconds: float) -> str:
    """把墙钟秒数格式化为适合终端快速扫描的 ``HH:MM:SS``。"""

    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ConsoleReporter:
    """以稳定文本展示长时间实验进度；非交互输出不会产生周期性噪声。"""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        heartbeat_seconds: float = 15.0,
    ) -> None:
        """保存输出流；只有真实 TTY 才启动心跳线程，日志重定向时保持简洁。"""

        self.stream = sys.stdout if stream is None else stream
        self.heartbeat_seconds = heartbeat_seconds

    def line(self, text: str = "") -> None:
        """立即打印一行，避免长任务因缓冲让操作者误以为卡死。"""

        print(text, file=self.stream, flush=True)

    def banner(self, title: str, subtitle: str | None = None) -> None:
        """打印流水线标题框，标明当前 run 或 batch 身份。"""

        width = max(60, len(title) + 4, len(subtitle or "") + 4)
        self.line("=" * width)
        self.line(f"  {title}")
        if subtitle:
            self.line(f"  {subtitle}")
        self.line("=" * width)

    def stage(self, current: int, total: int, label: str) -> None:
        """打印阶段进度条；阶段数固定，便于区分 Agent 与 Docker 耗时。"""

        completed = max(0, min(current - 1, total))
        bar = "█" * completed + "░" * (total - completed)
        self.line(f"\n[{bar}] 阶段 {current}/{total}  {label}")

    def task(self, current: int, total: int, instance_id: str) -> None:
        """打印批量求解的题目级进度条和当前 instance ID。"""

        width = 20
        filled = int(width * (current - 1) / total)
        bar = "█" * filled + "░" * (width - filled)
        self.line(f"\n[{bar}] 题目 {current}/{total}  {instance_id}")

    @contextmanager
    def activity(self, label: str) -> Iterator[None]:
        """包裹长操作并显示耗时；TTY 中每隔一段时间打印仍在运行的心跳。"""

        started = time.monotonic()
        stopped = Event()
        self.line(f"  ▶ {label}")

        def heartbeat() -> None:
            # Event.wait 同时承担可中断 sleep，结束时无需等待完整心跳周期。
            while not stopped.wait(self.heartbeat_seconds):
                elapsed = format_duration(time.monotonic() - started)
                self.line(f"  … {label}，已运行 {elapsed}")

        interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        worker = Thread(target=heartbeat, daemon=True) if interactive else None
        if worker is not None:
            worker.start()
        try:
            yield
        except Exception:
            elapsed = format_duration(time.monotonic() - started)
            self.line(f"  ✗ {label}失败（{elapsed}）")
            raise
        else:
            elapsed = format_duration(time.monotonic() - started)
            self.line(f"  ✓ {label}完成（{elapsed}）")
        finally:
            stopped.set()
            if worker is not None:
                worker.join(timeout=1)

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
        """打印对齐的纯文本表；无需 Rich 也能在 WSL、日志和 CI 中阅读。"""

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
