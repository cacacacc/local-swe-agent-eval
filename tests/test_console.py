"""验证终端可视化在普通日志流中的稳定文本结构。"""

from io import StringIO

from tracking.console import ConsoleReporter, format_duration


def test_format_duration_uses_fixed_width_clock() -> None:
    """长时间实验耗时必须使用固定宽度，避免汇总表随数值跳动。"""

    assert format_duration(0) == "00:00:00"
    assert format_duration(3661.9) == "01:01:01"


def test_reporter_prints_stage_task_and_aligned_table() -> None:
    """非 TTY 输出仍应包含阶段、题目进度和可读的汇总表。"""

    stream = StringIO()
    reporter = ConsoleReporter(stream)
    reporter.banner("Evaluation", "batch-001")
    reporter.stage(2, 4, "生成 predictions")
    reporter.task(3, 10, "owner__repo-3")
    reporter.table(("#", "status"), ((1, "resolved"), (2, "unresolved")))

    output = stream.getvalue()
    assert "Evaluation" in output
    assert "阶段 2/4" in output
    assert "题目 3/10" in output
    assert "resolved" in output and "unresolved" in output
