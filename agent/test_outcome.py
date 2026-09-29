"""把不同仓库的测试输出规范化为可比较的失败签名。

Docker 沙箱只负责可靠执行并保留输出；本模块再按实际 argv 识别 pytest、Django
unittest 与 SymPy ``bin/test`` 的失败摘要。调度器比较签名集合而不是只比较退出码，
因此 baseline 已有无关失败时，candidate 新增的失败仍能触发聚焦修复。
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
# unittest 的 subTest 参数有时会包含默认对象 repr，其中内存地址每个 Python
# 进程都不同。保留对象类型和其余稳定参数，只替换地址，避免 baseline/candidate
# 对同一测试生成不同签名。
_VOLATILE_HEX_ADDRESS = re.compile(r"(?<=\bat )0x[0-9a-fA-F]+\b")


@dataclass(frozen=True, slots=True)
class TestOutcome:
    """一次测试的规范化状态、失败测试集合与解析可信度。"""

    status: str
    failure_signatures: frozenset[str] = frozenset()
    parser: str = "none"
    reliable: bool = False


def parse_test_outcome(
    argv: Sequence[str],
    result: VisibleTestResult,
) -> TestOutcome:
    """按测试入口解析结果；无法可靠定位失败时明确返回不可信状态。

    成功退出无需失败摘要即可可靠判定。超时同样是确定状态，但没有可与另一次失败
    做集合差分的测试 ID；基础设施错误和未启动命令则完全不构成测试证据。
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
        # 非零退出但没有测试 ID 时，可能是收集错误、进程崩溃或未覆盖的输出格式。
        # 此时不能仅凭相同退出码声称 candidate 没有增加回归。
        reliable=bool(signatures),
    )


def _select_parser(argv: Sequence[str]) -> tuple[str, re.Pattern[str]]:
    """根据父进程生成的 argv 选择稳定解析器，未知命令使用 pytest 风格兜底。"""

    normalized = tuple(argv)
    if len(normalized) >= 2 and normalized[1] == "tests/runtests.py":
        return "django-unittest", _UNITTEST_FAILURE
    if len(normalized) >= 2 and normalized[1] == "bin/test":
        return "sympy-bin-test", _SYMPY_FAILURE
    if len(normalized) >= 3 and normalized[1:3] == ("-m", "pytest"):
        return "pytest", _PYTEST_FAILURE
    return "generic-pytest-summary", _PYTEST_FAILURE


def _normalize_signature(value: str) -> str:
    """移除进程相关地址并压缩空白，生成跨容器稳定的失败测试 ID。"""

    without_addresses = _VOLATILE_HEX_ADDRESS.sub("0xADDR", value)
    return " ".join(without_addresses.strip().split())
