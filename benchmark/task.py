"""定义暴露给求解 Agent 的安全版 SWE-bench 任务对象。

原始 SWE-bench 记录还可能包含参考补丁、隐藏测试补丁等评测专用字段。
本模块通过显式白名单只保留求解所需的四个字段，从数据结构层面降低答案泄漏风险。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


class TaskValidationError(ValueError):
    """当数据记录无法唯一、可复现地描述一道任务时抛出。"""


# 仓库必须采用 GitHub 的 ``owner/name`` 形式，避免任意 URL 或路径注入。
_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
# Git 支持使用短 SHA；7 到 40 位覆盖常用短 SHA 和完整 SHA-1。
_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{7,40}$")


@dataclass(frozen=True, slots=True)
class SWEbenchTask:
    """允许传入求解 Agent 的最小任务表示。

    ``patch``、``test_patch`` 等评测字段被刻意排除。保持数据类型足够小，
    可以让代码审查和自动测试更容易发现意外的数据泄漏。
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str

    def __post_init__(self) -> None:
        """在不可变 dataclass 创建后立即验证所有关键标识。"""

        # 先统一检查类型和空字符串，避免后续正则匹配产生含糊错误。
        values = {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
        }
        for field_name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise TaskValidationError(
                    f"{field_name} must be a non-empty string"
                )

        if not _REPOSITORY_PATTERN.fullmatch(self.repo):
            raise TaskValidationError(
                "repo must have the GitHub 'owner/name' form"
            )
        if not _COMMIT_PATTERN.fullmatch(self.base_commit):
            raise TaskValidationError(
                "base_commit must be a 7-40 character hexadecimal Git commit"
            )

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SWEbenchTask":
        """验证原始数据集记录，并且只复制白名单中的字段。

        即使 ``record`` 中含有 reference patch，本方法也不会将其保存到对象中。
        """

        required = (
            "instance_id",
            "repo",
            "base_commit",
            "problem_statement",
        )
        missing = [field for field in required if field not in record]
        if missing:
            raise TaskValidationError(
                f"missing required field(s): {', '.join(missing)}"
            )

        # 显式投影到 required 字段是防泄漏边界，不能改成 ``cls(**record)``。
        return cls(**{field: record[field] for field in required})

    def to_agent_payload(self) -> dict[str, str]:
        """返回允许进入 Agent 上下文的完整且唯一的数据载荷。"""

        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
        }
