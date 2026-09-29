"""Read legacy Agent plans and let the parent process generate repository-adapted visible test plans.

The model must not assemble or execute host test commands on its own. The parent
process generates the target-test and adjacent regression-test argv from the
repository layout and the patch, then hands them one by one to the offline Docker
sandbox. ``consume_test_plan`` exists only for compatibility and to clean up a
``.agent-test-plan.json`` that an old session may have left behind; commands in that
file no longer control execution, and the control file is deleted immediately after
being read, so it never pollutes the final patch.
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
    """One plan-read result; the rejection reason is also preserved as experiment data."""

    status: str
    target_argv: tuple[str, ...] = ()
    regression_argv: tuple[str, ...] = ()
    error: str | None = None
    origin: str = "agent"

    @property
    def accepted(self) -> bool:
        """Return ``True`` only when both structure and safety boundaries pass."""

        return self.status in {"accepted", "generated"}

    @property
    def requested(self) -> bool:
        """Distinguish a model that submitted no plan from one that submitted an invalid plan."""

        return self.status != "missing"

    @property
    def commands(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Return the target test and the adjacent regression test in a fixed order."""

        if not self.accepted:
            return ()
        return (
            ("target", self.target_argv),
            ("regression", self.regression_argv),
        )


def consume_test_plan(repository: Path | str) -> TestPlanRequest:
    """Read, validate, and delete the one-shot JSON test plan at the repository root.

    Deletion happens in a ``finally`` around parsing: even if the model writes an
    oversized, corrupt, or malicious plan, the control file never enters the candidate
    patch. The plan accepts only the two fixed argv fields, avoiding future extra
    fields being silently ignored by an old scheduler and creating a permission
    misunderstanding.
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
    """Deterministically generate two test commands from the repository type, modified paths, and existing test layout.

    The generator only selects test files that already exist in the task repository
    and never reads the SWE-bench hidden ``test_patch``. The target test takes the
    existing test module closest to the modified source file name with fail-fast;
    the regression test selects another test module in the same directory or with the
    deepest shared path, avoiding passing off the same command with ``-x`` removed as
    the adjacent regression. Django's regression command runs the target test app.
    Django uses the project's own ``tests/runtests.py``, SymPy uses ``bin/test``
    (which does not depend on pytest), and the remaining SWE-bench Python repositories
    use pytest. When no trustworthy test is found it returns ``missing``; the caller
    should still hand that fact to Verification rather than treating it as a hard gate.
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
    """Extract target paths from the patch and mark this round's newly created files for the planner to exclude."""

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
    """Enumerate existing Python tests in supported repositories, keeping the order sorted for reproducibility."""

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
    # Astropy, scikit-learn, xarray, SymPy, and others keep tests inside the source package.
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
    """Select a target file and a different adjacent regression file based on the modified paths."""

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
        # The first segment is usually just the top-level package name (astropy,
        # sklearn, sphinx, etc.); keeping it would give nearly all tests the same
        # score and reduce selection to lexicographic order, so only the more specific
        # inner path is used.
        for part in source.with_suffix("").parts[1:]:
            for token in re.findall(r"[a-z0-9]+", part.lower()):
                if len(token) >= 3 and token not in ignored:
                    tokens.add(token)
                    # Package paths often use version-suffixed names like sqlite3,
                    # while test modules usually drop the digits; keep the trailing-
                    # digits-stripped form too, so a direct corresponding test is not
                    # missed due to differing naming conventions.
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
        # On ties prefer shorter, more specific paths; finally use string order to
        # eliminate filesystem differences.
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

    # The list is already sorted by path; max keeps the first on ties, so selection
    # remains reproducible across filesystems.
    regression = max(
        (path for path in test_files if path != target),
        key=adjacent_score,
        default=None,
    )
    return target, regression


def _django_test_prefixes(source_paths: list[Path]) -> set[str]:
    """Map Django source subsystems to test apps, avoiding module misselection based only on a generic file name."""

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
    """Build the two commands for pytest repositories: the target file and a different adjacent scope."""

    regression_path = regression
    if regression_path is None and target.parent != Path("."):
        # When a tiny repository has a single test file, fall back to its directory;
        # the argv still differs from the target file and can cover same-directory
        # fixture/collection behavior. The repository root is not used as a fallback,
        # to avoid accidentally running the full test suite.
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
    """Generate project-native test commands for SymPy images that do not preinstall pytest.

    The old SWE-bench SymPy images only guarantee ``bin/test`` is available.
    ``--no-colors`` keeps terminal control characters out of the persisted log, and
    ``-C`` disables the runner's own cache so the baseline and candidate are not
    affected by in-repository cache within one experiment; the parent's own result
    cache remains strictly keyed by the patch digest.
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
    """Convert a path under ``tests/`` into a dotted label accepted by Django's own runner."""

    if not target.parts or target.parts[0] != "tests" or len(target.parts) < 2:
        return None
    relative = target.relative_to("tests").with_suffix("")
    target_label = ".".join(relative.parts)
    if len(relative.parts) >= 2:
        regression_label = relative.parts[0]
    elif regression is not None and regression.parts[:1] == ("tests",):
        # A few Django root-level tests (such as tests/test_sqlite.py) have no app to
        # widen to; in that case use the other test module selected by the planner,
        # still ensuring the two commands cover different scopes.
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
    """Clean up only the fixed-name control path inside the isolated repository, without following a directory symlink."""

    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    else:
        # A directory can only have been created by the current Agent in the one-shot
        # workspace; precise removal preserves the "control file never enters the
        # patch" safety boundary without extending to the rest of the repository.
        shutil.rmtree(path)


def _validate_plan(value: Any) -> str | None:
    """Return the first validation error; ``None`` means the argv can be handed directly to subprocess."""

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
    """Validate one shell-free test argv, preserving the field name in errors."""

    if not isinstance(argv, list) or not argv or len(argv) > _MAX_ARGUMENTS:
        return f"{field} must contain between 1 and {_MAX_ARGUMENTS} string items"
    for item in argv:
        if not isinstance(item, str) or not item or len(item) > _MAX_ARGUMENT_CHARS:
            return f"every {field} item must be a non-empty bounded string"
        if any(ord(character) < 32 and character not in {"\t"} for character in item):
            return f"{field} items cannot contain control characters"

    executable = Path(argv[0])
    # Tests start under the container's fixed /testbed; forbid absolute paths and
    # parent-directory escapes so the model cannot pick host tools outside the image.
    # Shell, network, and package-management entries are unnecessary even in an
    # offline container.
    if executable.is_absolute() or ".." in executable.parts:
        return "test executable must stay relative to /testbed or use PATH"
    if executable.name.lower() in _FORBIDDEN_EXECUTABLES:
        return f"test executable is forbidden: {executable.name}"
    return None
