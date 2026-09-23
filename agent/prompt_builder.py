"""Render one fixed, auditable prompt for every SWE-bench task."""

from __future__ import annotations

from pathlib import Path
from string import Template

from benchmark.task import SWEbenchTask


class PromptTemplateError(ValueError):
    """Raised when a prompt template is missing required task placeholders."""


class PromptBuilder:
    """Load a template once and render only approved task fields."""

    REQUIRED_PLACEHOLDERS = (
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
    )

    def __init__(self, template_text: str) -> None:
        normalized = template_text.replace("\r\n", "\n").replace("\r", "\n")
        if not normalized.strip():
            raise PromptTemplateError("prompt template cannot be empty")
        missing = [
            name
            for name in self.REQUIRED_PLACEHOLDERS
            if f"${{{name}}}" not in normalized
        ]
        if missing:
            raise PromptTemplateError(
                f"prompt template is missing placeholder(s): {', '.join(missing)}"
            )
        self._template = Template(normalized)

    @classmethod
    def from_file(cls, path: Path | str) -> "PromptBuilder":
        source = Path(path)
        try:
            return cls(source.read_text(encoding="utf-8"))
        except OSError as error:
            raise PromptTemplateError(f"cannot read prompt template {source}: {error}") from error

    def build(self, task: SWEbenchTask) -> str:
        rendered = self._template.substitute(task.to_agent_payload())
        return f"{rendered.rstrip()}\n"
