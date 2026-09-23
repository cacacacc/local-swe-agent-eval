"""使用固定模板为每道 SWE-bench 任务生成可审计 Prompt。

模板在构造时一次性验证，渲染时只接收 ``SWEbenchTask`` 的安全字段，确保不同
任务之间只有题目数据变化，而实验指令保持一致。
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from benchmark.task import SWEbenchTask


class PromptTemplateError(ValueError):
    """当 Prompt 模板为空、不可读或缺少必要占位符时抛出。"""


class PromptBuilder:
    """加载并验证模板，只使用批准的任务字段完成渲染。"""

    REQUIRED_PLACEHOLDERS = (
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
    )

    def __init__(self, template_text: str) -> None:
        """规范化换行符并确认四个任务字段均有显式占位符。"""

        # 统一成 LF，避免 Windows/WSL 行尾差异改变 prompt hash。
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
        """从 UTF-8 文本文件读取模板，并把文件系统错误转换成领域错误。"""

        source = Path(path)
        try:
            return cls(source.read_text(encoding="utf-8"))
        except OSError as error:
            raise PromptTemplateError(f"cannot read prompt template {source}: {error}") from error

    def build(self, task: SWEbenchTask) -> str:
        """渲染单个任务，并保证结果恰好以一个换行符结束。"""

        # ``to_agent_payload`` 是唯一的数据入口，因此评测专用字段无法被替换进模板。
        rendered = self._template.substitute(task.to_agent_payload())
        return f"{rendered.rstrip()}\n"
