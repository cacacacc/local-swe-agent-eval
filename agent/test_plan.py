"""读取旧 Agent 计划，并由父进程生成仓库适配的可见测试计划。

模型不得自行拼接或执行宿主测试命令。父进程根据仓库布局和 patch 生成目标测试
与相邻回归测试 argv，再依次交给无网络 Docker 沙箱。``consume_test_plan`` 仅为
兼容和清理旧会话可能遗留的 ``.agent-test-plan.json``；其中命令不再控制执行，
控制文件在读取后立即删除，因此不会污染最终补丁。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
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
    origin: str = "agent"

    @property
    def accepted(self) -> bool:
        """仅当结构与安全边界全部通过时返回 ``True``。"""

        return self.status in {"accepted", "generated"}

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


def generate_repository_test_plan(
    repository: Path | str,
    *,
    repo: str,
    patch: str,
) -> TestPlanRequest:
    """根据仓库类型、修改路径和现有测试布局确定性生成两条测试命令。

    生成器只选择任务仓库中已经存在的测试文件，绝不读取 SWE-bench 隐藏
    ``test_patch``。目标测试取与修改源码文件名最接近的现有测试模块并采用
    fail-fast；回归测试选择同目录或共同路径最深的另一个测试模块，避免用取消
    ``-x`` 的同一命令冒充相邻回归。Django 的回归命令运行目标测试 app。
    Django 使用项目自己的 ``tests/runtests.py``，SymPy 使用不依赖 pytest 的
    ``bin/test``，其余 SWE-bench Python 仓库使用 pytest。无法找到可信测试时
    返回 ``missing``，调用方仍应把这一事实交给 Verification，而不能把它当作
    硬门禁。
    """

    repository_path = Path(repository).resolve()
    changed_paths, new_paths = _patch_paths(patch)
    source_paths = [path for path in changed_paths if path not in new_paths]
    if not source_paths:
        return TestPlanRequest(
            status="missing",
            error="parent planner found no modified existing source path",
            origin="parent",
        )

    test_files = _repository_test_files(repository_path, repo)
    selected = _select_test_files(test_files, source_paths, repo=repo)
    if selected is None:
        return TestPlanRequest(
            status="missing",
            error=f"parent planner found no repository-visible test for {repo}",
            origin="parent",
        )

    target, regression = selected
    if repo == "django/django":
        commands = _django_commands(target, regression)
    elif repo == "sympy/sympy":
        commands = _sympy_commands(target, regression)
    else:
        commands = _pytest_commands(target, regression)
    if commands is None:
        return TestPlanRequest(
            status="missing",
            error=f"parent planner could not adapt test path {target.as_posix()}",
            origin="parent",
        )
    target_argv, regression_argv = commands
    return TestPlanRequest(
        status="generated",
        target_argv=target_argv,
        regression_argv=regression_argv,
        origin="parent",
    )


def _patch_paths(patch: str) -> tuple[list[Path], set[Path]]:
    """提取补丁中的目标路径，并标记本轮新建文件供计划器排除。"""

    changed: list[Path] = []
    newly_created: set[Path] = set()
    for section in re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE):
        match = re.match(r"diff --git a/(.+?) b/(.+?)\n", section)
        if match is None:
            continue
        path = Path(match.group(2))
        changed.append(path)
        if "new file mode " in section:
            newly_created.add(path)
    return changed, newly_created


def _repository_test_files(repository: Path, repo: str) -> list[Path]:
    """枚举受支持仓库的既有 Python 测试，保持排序以保证可复现。"""

    roots = ("tests",) if repo == "django/django" else ("tests", "testing")
    candidates: set[Path] = set()
    for root_name in roots:
        root = repository / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if path.is_file() and (
                path.name.startswith("test_")
                or path.name.endswith("_test.py")
                or root_name == "testing"
            ):
                candidates.add(path.relative_to(repository))
    # Astropy、scikit-learn、xarray、SymPy 等把测试放在源码包内部。
    if not candidates:
        for path in repository.rglob("test_*.py"):
            if path.is_file() and ".git" not in path.parts:
                candidates.add(path.relative_to(repository))
    return sorted(candidates, key=lambda path: path.as_posix())


def _select_test_files(
    test_files: list[Path],
    source_paths: list[Path],
    *,
    repo: str,
) -> tuple[Path, Path | None] | None:
    """按修改路径选择目标文件和一个不同的相邻回归文件。"""

    ignored = {
        "__init__",
        "base",
        "common",
        "core",
        "lib",
        "main",
        "models",
        "src",
        "test",
        "tests",
        "testing",
        "utils",
    }
    tokens: set[str] = set()
    for source in source_paths:
        # 首段通常只是顶层包名（astropy、sklearn、sphinx 等），保留它会让几乎
        # 所有测试同分并退化成字典序选择，因此只使用更具体的内部路径。
        for part in source.with_suffix("").parts[1:]:
            for token in re.findall(r"[a-z0-9]+", part.lower()):
                if len(token) >= 3 and token not in ignored:
                    tokens.add(token)
                    # 包路径常用 sqlite3 等带版本后缀名称，而测试模块通常省略数字；
                    # 同时保留去尾数字形式，避免因命名惯例差异错过直接对应测试。
                    without_version = token.rstrip("0123456789")
                    if len(without_version) >= 3:
                        tokens.add(without_version)
    if not tokens:
        return None

    if repo == "django/django":
        preferred = _django_test_prefixes(source_paths)
        preferred_files = [
            path
            for path in test_files
            if any(prefix in path.parts for prefix in preferred)
        ]
        if preferred_files:
            test_files = preferred_files

    def score(path: Path) -> tuple[int, int]:
        text = path.with_suffix("").as_posix().lower()
        matched = sum(1 for token in tokens if token in text)
        exact_stem = int(any(path.stem == f"test_{token}" for token in tokens))
        # 同分时优先更短、更具体的路径，最后用字符串顺序消除文件系统差异。
        return (exact_stem * 100 + matched * 10, -len(path.parts))

    target = max(test_files, key=score, default=None)
    if target is None or score(target)[0] <= 0:
        return None

    def adjacent_score(path: Path) -> tuple[int, int, int, int]:
        common_depth = 0
        for left, right in zip(path.parts, target.parts):
            if left != right:
                break
            common_depth += 1
        relevance, compactness = score(path)
        return (
            int(path.parent == target.parent),
            common_depth,
            relevance,
            compactness,
        )

    # 列表已按路径排序；max 在同分时保留第一个，因而跨文件系统仍可复现。
    regression = max(
        (path for path in test_files if path != target),
        key=adjacent_score,
        default=None,
    )
    return target, regression


def _django_test_prefixes(source_paths: list[Path]) -> set[str]:
    """把 Django 源码子系统映射到测试 app，避免仅凭通用文件名误选模块。"""

    mappings = {
        ("django", "forms"): "forms_tests",
        ("django", "template"): "template_tests",
        ("django", "db", "migrations"): "migrations",
        ("django", "db", "models", "fields"): "model_fields",
        ("django", "contrib", "auth"): "auth_tests",
        ("django", "contrib", "admin"): "admin_views",
    }
    prefixes: set[str] = set()
    for source in source_paths:
        parts = source.parts
        for prefix, test_app in mappings.items():
            if parts[: len(prefix)] == prefix:
                prefixes.add(test_app)
    return prefixes


def _pytest_commands(
    target: Path,
    regression: Path | None,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """为 pytest 仓库构造目标文件和不同相邻范围的两条命令。"""

    regression_path = regression
    if regression_path is None and target.parent != Path("."):
        # 极小仓库只有一个测试文件时退回其目录；argv 仍与目标文件不同，并能覆盖
        # 同目录 fixture/收集行为。仓库根目录不作为回退，避免意外跑全量测试。
        regression_path = target.parent
    if regression_path is None:
        return None

    return (
        ("python", "-m", "pytest", "-x", target.as_posix()),
        ("python", "-m", "pytest", regression_path.as_posix()),
    )


def _sympy_commands(
    target: Path,
    regression: Path | None,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """为不预装 pytest 的 SymPy 镜像生成项目原生测试命令。

    SWE-bench 的旧版 SymPy 镜像只保证 ``bin/test`` 可用。``--no-colors`` 让持久化
    日志不含终端控制字符，``-C`` 关闭 runner 自身缓存，防止基线与候选在同一实验
    中受到仓库内部缓存影响；父进程自己的结果缓存仍由 patch digest 严格控制。
    """

    regression_path = regression
    if regression_path is None and target.parent != Path("."):
        regression_path = target.parent
    if regression_path is None:
        return None

    prefix = ("python", "bin/test", "--no-colors", "-C")
    return (
        (*prefix, target.as_posix()),
        (*prefix, regression_path.as_posix()),
    )


def _django_commands(
    target: Path,
    regression: Path | None,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """把 ``tests/`` 下路径转换成 Django 自带 runner 接受的 dotted label。"""

    if not target.parts or target.parts[0] != "tests" or len(target.parts) < 2:
        return None
    relative = target.relative_to("tests").with_suffix("")
    target_label = ".".join(relative.parts)
    if len(relative.parts) >= 2:
        regression_label = relative.parts[0]
    elif regression is not None and regression.parts[:1] == ("tests",):
        # 少数 Django 根级测试（如 tests/test_sqlite.py）没有 app 可扩大；此时
        # 使用规划器选出的另一个测试模块，仍保证两条命令覆盖不同范围。
        regression_label = ".".join(
            regression.relative_to("tests").with_suffix("").parts
        )
    else:
        return None
    return (
        (
            "python",
            "tests/runtests.py",
            target_label,
            "--failfast",
            "--verbosity",
            "0",
        ),
        ("python", "tests/runtests.py", regression_label, "--verbosity", "0"),
    )


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
