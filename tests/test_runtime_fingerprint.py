"""Verify that the runtime fingerprint uses the current experiment interpreter rather than host command aliases."""

from pathlib import Path
import sys

from experiment.runtime_fingerprint import RuntimeFingerprintCollector


def test_python_version_uses_current_interpreter(tmp_path: Path, monkeypatch) -> None:
    """When there is no bare `python` command, fingerprint collection should still reuse the interpreter that launched the batch."""

    prompt = tmp_path / "prompt.txt"
    prompt.write_text("test prompt", encoding="utf-8")
    collector = RuntimeFingerprintCollector(
        project_root=tmp_path,
        prompt_path=prompt,
        model_name="test-model",
        base_url="http://localhost:11434",
    )
    commands: list[list[str]] = []

    def fake_run(command):
        commands.append(list(command))
        return ""

    monkeypatch.setattr(collector, "_run", fake_run)
    monkeypatch.setattr(
        collector,
        "_ollama_model",
        lambda: {"name": "test-model", "digest": "abc"},
    )

    collector.collect()

    assert [
        sys.executable,
        "-c",
        "import platform; print(platform.python_version())",
    ] in commands
