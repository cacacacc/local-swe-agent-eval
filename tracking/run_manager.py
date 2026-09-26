"""创建不可静默覆盖的运行目录，并持久化可观察实验数据。

每次运行会保存 metadata、prompt、agent 日志、trajectory、Git patch、测试输出
和结果摘要。这里记录的是可验证行为，不记录或推测模型隐藏思维过程。
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping, Sequence

from benchmark.task import SWEbenchTask


class RunArtifactError(RuntimeError):
    """当运行产物无法在不丢失数据的前提下创建时抛出。"""


# instance ID 来自外部数据集，必须先转换为安全的单层目录名。
_SAFE_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _utc_now() -> str:
    """生成带时区的 UTC 时间戳，避免不同机器本地时区造成歧义。"""

    return datetime.now(timezone.utc).isoformat()


class RunSession:
    """单道任务的一次运行会话；完成后不可再次 finalize。"""

    def __init__(
        self,
        path: Path,
        task: SWEbenchTask,
        metadata: dict[str, Any],
        started_monotonic: float,
        patch_base_commit: str,
    ) -> None:
        """保存运行上下文；单调时钟专用于准确计算耗时。"""

        self.path = path
        self.task = task
        self._metadata = metadata
        self._started_monotonic = started_monotonic
        self._patch_base_commit = patch_base_commit
        self._finished = False

    def collect_patch(self, repository: Path | str) -> str:
        """收集相对任务基线的 committed、tracked 与 untracked 修改。

        ``git diff`` 默认不会包含未跟踪文件，因此先用 ``--intent-to-add``
        将它们标记为“计划加入”，但不会真正创建 commit。diff 必须显式以任务的
        隔离仓库的单提交基线为左侧；Agent 可能自行 commit，若只比较 working
        tree 会把已经提交的有效修复误判为空 patch。上游原始 SHA 只用于审计，
        不能作为隔离仓库中并不存在的对象参与 diff。
        """

        repository_path = Path(repository).resolve()
        self._run_git(repository_path, "add", "--intent-to-add", "--all")
        return self._run_git(
            repository_path,
            "diff",
            "--binary",
            "--no-ext-diff",
            self._patch_base_commit,
            "--",
        )

    def finalize(
        self,
        *,
        exit_code: int,
        agent_log: str,
        test_output: str,
        events: Sequence[Mapping[str, Any]],
        patch: str,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        """原子写入本次运行的最终产物，并将会话标记为完成。

        ``exit_code == 0`` 只表示 Agent 进程正常结束，不等价于 SWE-bench issue
        已解决；官方评测结果因此保持为 ``None``，等待 harness 后续填写。
        """

        if self._finished:
            raise RunArtifactError(f"run has already been finalized: {self.path}")

        end_time = _utc_now()
        runtime_seconds = round(time.monotonic() - self._started_monotonic, 6)
        # 124 沿用 GNU timeout 的约定，单独分类后才能准确计算 timeout rate。
        status = "completed" if exit_code == 0 else "timeout" if exit_code == 124 else "failed"
        patch_line_count = sum(
            1
            for line in patch.splitlines()
            if (line.startswith("+") and not line.startswith("+++"))
            or (line.startswith("-") and not line.startswith("---"))
        )

        # 先写详细产物，再更新 metadata；这样异常时仍能保留尽可能多的证据。
        self._write_text("agent.log", agent_log)
        self._write_text("test_output.log", test_output)
        self._write_text("patch.diff", patch)
        self._write_json(
            "trajectory.json",
            {"schema_version": 1, "events": list(events)},
        )
        self._write_json(
            "result.json",
            {
                "schema_version": 1,
                "run_status": status,
                "agent_exit_code": exit_code,
                "patch_generated": bool(patch.strip()),
                "patch_line_count": patch_line_count,
                "metrics": dict(metrics) if metrics is not None else {},
                "official_evaluation": None,
            },
        )

        self._metadata.update(
            {
                "end_time": end_time,
                "runtime_seconds": runtime_seconds,
                "status": status,
                "event_count": len(events),
                "metrics": dict(metrics) if metrics is not None else {},
            }
        )
        self._write_json("metadata.json", self._metadata)
        self._finished = True

    def _write_text(self, name: str, content: str) -> None:
        """先写同目录临时文件，再 replace，避免留下半写入文件。"""

        destination = self.path / name
        temporary = self.path / f".{name}.tmp"
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)

    def _write_json(self, name: str, content: Mapping[str, Any]) -> None:
        """以稳定键顺序和 UTF-8 格式保存便于审计的 JSON。"""

        serialized = json.dumps(
            content,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        self._write_text(name, f"{serialized}\n")

    @staticmethod
    def _run_git(repository: Path, *arguments: str) -> str:
        """运行补丁收集所需的 Git 命令，并保留失败细节。"""

        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no output"
            raise RunArtifactError(
                f"git command failed ({result.returncode}): "
                f"git {' '.join(arguments)}\n{detail}"
            )
        return result.stdout


class RunManager:
    """在 Agent 执行前创建运行目录并写入初始 metadata。"""

    def __init__(self, runs_root: Path | str) -> None:
        """保存解析后的运行根目录，避免后续受当前工作目录变化影响。"""

        self.runs_root = Path(runs_root).resolve()

    def start(
        self,
        task: SWEbenchTask,
        *,
        phase: str,
        agent: str,
        model: str,
        prompt: str,
        configuration: Mapping[str, Any] | None = None,
        patch_base_commit: str | None = None,
    ) -> RunSession:
        """启动新会话并立即持久化 prompt 与初始元数据。

        目录使用 ``exist_ok=False``：相同 instance ID 的旧结果不会被新运行
        静默覆盖。若要重复实验，调用方必须提供新的运行根目录或 run ID。
        """

        safe_instance_id = _SAFE_PATH_COMPONENT.sub("_", task.instance_id)
        run_path = self.runs_root / safe_instance_id
        try:
            run_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise RunArtifactError(
                f"run directory already exists; refusing to overwrite: {run_path}"
            ) from error
        except OSError as error:
            raise RunArtifactError(f"cannot create run directory {run_path}: {error}") from error

        start_time = _utc_now()
        # 在 Agent 启动前就落盘，进程崩溃时仍能知道任务和启动配置。
        metadata = {
            "schema_version": 1,
            "instance_id": task.instance_id,
            "repository": task.repo,
            "base_commit": task.base_commit,
            "experiment_phase": phase,
            "agent": agent,
            "model": model,
            "start_time": start_time,
            "end_time": None,
            "runtime_seconds": None,
            "status": "running",
            "configuration": dict(configuration) if configuration is not None else None,
        }
        # 旧调用方仍可传入普通 checkout，此时原始 base commit 继续有效；新的隔离
        # RepositoryManager 会显式传入重新初始化后的 workspace baseline。
        effective_patch_base = patch_base_commit or task.base_commit
        metadata["workspace_base_commit"] = effective_patch_base
        session = RunSession(
            run_path,
            task,
            metadata,
            time.monotonic(),
            effective_patch_base,
        )
        session._write_json("metadata.json", metadata)
        session._write_text("prompt.txt", prompt)
        return session
