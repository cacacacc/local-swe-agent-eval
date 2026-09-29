"""Collect runtime versions, content hashes, and Ollama model identity for local experiments.

The configuration fingerprint only proves the YAML parameters match; it cannot prove
that two runs used the same binary, Git commit, or model file. This module normalizes
that changeable external state and computes a SHA-256 again, saving it in each task's
metadata so the final report can trace the actual runtime environment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class RuntimeFingerprintError(RuntimeError):
    """Raised when a required version or the local model identity cannot be collected reliably."""


def _sha256_file(path: Path) -> str:
    """Compute a file's SHA-256 in binary mode so platform newline conversion cannot change the read result."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RuntimeFingerprintCollector:
    """Build a stable runtime fingerprint from local commands and the Ollama loopback API."""

    def __init__(
        self,
        *,
        project_root: Path | str,
        prompt_path: Path | str,
        model_name: str,
        base_url: str,
        command_timeout_seconds: int = 30,
    ) -> None:
        """Store the collection targets and re-confirm that the model API may only point at the local machine."""

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
        """Collect all required fields and compute the overall fingerprint over the normalized result."""

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
                [
                    sys.executable,
                    "-c",
                    "import platform; print(platform.python_version())",
                ]
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
        """Read the exact digest from `/api/tags`, refusing to record the model by a mutable tag alone."""

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
        """Run a read-only version command and translate failures uniformly into contextual domain errors."""

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
