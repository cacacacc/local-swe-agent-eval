"""读取并校验 Agent 交给调度器的结构化可见测试计划。

模型不得自行拼接或执行宿主测试命令。实现/修复会话只可在仓库根目录写入
``.agent-test-plan.json``。计划必须分别声明目标测试和相邻回归测试的 argv，
调度器依次把两条命令传给无网络 Docker 沙箱。计划文件在读取后立即删除，
因此不会污染最终补丁。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any


TEST_PLAN_FILENAME = ".agent-test-plan.json"
_MAX_PLAN_BYTES = 16_384
_MAX_ARGUMENTS = 64
_MAX_ARGUMENT_CHARS = 4_096
_FORBIDDEN_EXECUTABLES = {
    "bash",
    "curl",
    "docker",
    "git",
    "pip",
    "pip3",
    "sh",
    "sudo",
    "wget",
    "zsh",
}


@dataclass(frozen=True, slots=True)
class TestPlanRequest:
    """一次计划读取结果；拒绝原因也作为实验数据保留。"""

    status: str
    target_argv: tuple[str, ...] = ()
    regression_argv: tuple[str, ...] = ()
    error: str | None = None

    @property
    def accepted(self) -> bool:
        """仅当结构与安全边界全部通过时返回 ``True``。"""

        return self.status == "accepted"

    @property
    def requested(self) -> bool:
        """区分模型未提交计划与提交了无效计划。"""

        return self.status != "missing"

    @property
    def commands(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """按固定顺序返回目标测试与相邻回归测试。"""

        if not self.accepted:
            return ()
        return (
            ("target", self.target_argv),
            ("regression", self.regression_argv),
        )


def consume_test_plan(repository: Path | str) -> TestPlanRequest:
    """读取、校验并删除仓库根目录中的一次性 JSON 测试计划。

    删除发生在解析前后的 ``finally`` 中：即使模型写入超大、损坏或恶意计划，
    该控制文件也不会进入 candidate patch。计划只接受两个固定 argv 字段，避免
    日后新增字段被旧调度器静默忽略并产生权限误解。
    """

    repository_path = Path(repository).resolve()
    plan_path = repository_path / TEST_PLAN_FILENAME
    if not plan_path.exists():
        return TestPlanRequest(status="missing")
    if plan_path.is_symlink() or not plan_path.is_file():
        _remove_control_path(plan_path)
        return TestPlanRequest(
            status="rejected",
            error="test plan must be a regular file",
        )

    try:
        if plan_path.stat().st_size > _MAX_PLAN_BYTES:
            return TestPlanRequest(
                status="rejected",
                error="test plan exceeds 16384 bytes",
            )
        try:
            value: Any = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            return TestPlanRequest(
                status="rejected",
                error=f"cannot parse test plan: {error}",
            )
        error = _validate_plan(value)
        if error is not None:
            return TestPlanRequest(status="rejected", error=error)
        return TestPlanRequest(
            status="accepted",
            target_argv=tuple(value["target_argv"]),
            regression_argv=tuple(value["regression_argv"]),
        )
    finally:
        _remove_control_path(plan_path)


def _remove_control_path(path: Path) -> None:
    """只清理隔离仓库内固定名称的控制路径，不跟随目录 symlink。"""

    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    else:
        # 目录只可能由当前 Agent 在一次性 workspace 中创建；精确删除保留“控制
        # 文件绝不进入 patch”的安全边界，同时不会扩大到仓库其他内容。
        shutil.rmtree(path)


def _validate_plan(value: Any) -> str | None:
    """返回首个验证错误；``None`` 表示 argv 可直接交给 subprocess。"""

    expected_fields = {"target_argv", "regression_argv"}
    if not isinstance(value, dict) or set(value) != expected_fields:
        return (
            "test plan must contain only target_argv and regression_argv"
        )
    for field in ("target_argv", "regression_argv"):
        error = _validate_argv(value.get(field), field)
        if error is not None:
            return error
    if value["target_argv"] == value["regression_argv"]:
        return "target and regression argv must be different"
    return None


def _validate_argv(argv: Any, field: str) -> str | None:
    """校验一条无 shell 的测试 argv，并在错误中保留字段名称。"""

    if not isinstance(argv, list) or not argv or len(argv) > _MAX_ARGUMENTS:
        return f"{field} must contain between 1 and {_MAX_ARGUMENTS} string items"
    for item in argv:
        if not isinstance(item, str) or not item or len(item) > _MAX_ARGUMENT_CHARS:
            return f"every {field} item must be a non-empty bounded string"
        if any(ord(character) < 32 and character not in {"\t"} for character in item):
            return f"{field} items cannot contain control characters"

    executable = Path(argv[0])
    # 测试在容器固定的 /testbed 下启动；禁止绝对路径和父目录逃逸，防止模型选择
    # 镜像外的宿主工具。shell、网络和包管理入口即使在无网络容器中也没有必要。
    if executable.is_absolute() or ".." in executable.parts:
        return "test executable must stay relative to /testbed or use PATH"
    if executable.name.lower() in _FORBIDDEN_EXECUTABLES:
        return f"test executable is forbidden: {executable.name}"
    return None
