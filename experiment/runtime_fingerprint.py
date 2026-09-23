"""采集本地实验运行时版本、内容哈希和 Ollama 模型身份。

配置 fingerprint 只能证明 YAML 参数一致；它不能证明两次运行使用了同一个二进制、
Git 提交或模型文件。本模块把这些可变化的外部状态规范化并再次计算 SHA-256，供每道
任务的 metadata 保存，从而让最终报告可以追溯实际运行环境。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class RuntimeFingerprintError(RuntimeError):
    """当必需版本或本地模型身份无法可靠采集时抛出。"""


def _sha256_file(path: Path) -> str:
    """以二进制方式计算文件 SHA-256，避免平台换行转换改变读取结果。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RuntimeFingerprintCollector:
    """从本机命令和 Ollama loopback API 构建稳定的运行时指纹。"""

    def __init__(
        self,
        *,
        project_root: Path | str,
        prompt_path: Path | str,
        model_name: str,
        base_url: str,
        command_timeout_seconds: int = 30,
    ) -> None:
        """保存采集目标，并再次限制模型 API 只能指向本机。"""

        self.project_root = Path(project_root).resolve()
        self.prompt_path = Path(prompt_path).resolve()
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.command_timeout_seconds = command_timeout_seconds
        parsed = urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise RuntimeFingerprintError("Ollama fingerprint endpoint must be local")

    def collect(self) -> dict[str, Any]:
        """采集所有必需字段，并对规范化结果计算总 fingerprint。"""

        values: dict[str, Any] = {
            "schema_version": 1,
            "project_git_commit": self._run(
                ["git", "-C", str(self.project_root), "rev-parse", "HEAD"]
            ),
            "project_git_dirty": bool(
                self._run(
                    ["git", "-C", str(self.project_root), "status", "--porcelain"]
                )
            ),
            "prompt_sha256": _sha256_file(self.prompt_path),
            "claude_version": self._run(["claude", "--version"]),
            "docker_version": self._run(
                ["docker", "version", "--format", "{{.Client.Version}}/{{.Server.Version}}"]
            ),
            "python_version": self._run(
                ["python", "-c", "import platform; print(platform.python_version())"]
            ),
            "model": self._ollama_model(),
        }
        canonical = json.dumps(
            values,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        values["runtime_fingerprint"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        return values

    def _ollama_model(self) -> dict[str, Any]:
        """从 `/api/tags` 读取精确 digest，拒绝仅按易变 tag 记录模型。"""

        request = Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=self.command_timeout_seconds) as response:
                payload = json.load(response)
        except (OSError, URLError, json.JSONDecodeError) as error:
            raise RuntimeFingerprintError(f"cannot query local Ollama: {error}") from error

        models = payload.get("models") if isinstance(payload, Mapping) else None
        if not isinstance(models, list):
            raise RuntimeFingerprintError("Ollama /api/tags returned no model list")
        for model in models:
            if not isinstance(model, Mapping):
                continue
            names = {model.get("name"), model.get("model")}
            if self.model_name in names:
                digest = model.get("digest")
                if not isinstance(digest, str) or not digest:
                    raise RuntimeFingerprintError(
                        f"Ollama model {self.model_name} has no digest"
                    )
                details = model.get("details")
                return {
                    "name": self.model_name,
                    "digest": digest,
                    "size": model.get("size"),
                    "details": dict(details) if isinstance(details, Mapping) else None,
                }
        raise RuntimeFingerprintError(f"Ollama model is not installed: {self.model_name}")

    def _run(self, command: Sequence[str]) -> str:
        """执行只读版本命令，并把失败统一转换为带上下文的领域错误。"""

        try:
            result = subprocess.run(
                list(command),
                cwd=self.project_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.command_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeFingerprintError(
                f"cannot execute {' '.join(command)}: {error}"
            ) from error
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no output"
            raise RuntimeFingerprintError(
                f"command failed ({result.returncode}): {' '.join(command)}: {detail}"
            )
        return result.stdout.strip()
