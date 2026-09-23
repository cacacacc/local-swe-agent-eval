"""加载 SWE-bench 任务，同时阻止参考答案进入 Agent 上下文。

支持本地 JSON、JSONL 和可选的 Hugging Face 数据源。无论输入来自哪里，
最终都必须经过 :class:`SWEbenchTask` 的字段白名单与格式验证。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .task import SWEbenchTask, TaskValidationError


class DatasetFormatError(ValueError):
    """当任务集合格式错误或内部不一致时抛出。"""


class SWEbenchLoader(Sequence[SWEbenchTask]):
    """保持输入顺序且不含重复 instance ID 的安全任务集合。"""

    def __init__(self, tasks: Iterable[SWEbenchTask]) -> None:
        # tuple 保证加载完成后顺序稳定，字典则提供 O(1) 的按 ID 查询。
        self._tasks = tuple(tasks)
        self._by_id: dict[str, SWEbenchTask] = {}
        for task in self._tasks:
            if task.instance_id in self._by_id:
                raise DatasetFormatError(
                    f"duplicate instance_id: {task.instance_id}"
                )
            self._by_id[task.instance_id] = task

    @classmethod
    def from_records(
        cls, records: Iterable[Mapping[str, Any]]
    ) -> "SWEbenchLoader":
        """将一组原始记录转换成经过安全验证的任务集合。"""

        tasks: list[SWEbenchTask] = []
        for index, record in enumerate(records):
            try:
                tasks.append(SWEbenchTask.from_record(record))
            except (TaskValidationError, TypeError) as error:
                raise DatasetFormatError(
                    f"invalid record at index {index}: {error}"
                ) from error
        return cls(tasks)

    @classmethod
    def from_json(cls, path: Path | str) -> "SWEbenchLoader":
        """从顶层为数组的 UTF-8 JSON 文件加载任务。"""

        source = Path(path)
        try:
            content = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetFormatError(f"cannot read JSON dataset {source}: {error}") from error

        if not isinstance(content, list):
            raise DatasetFormatError("JSON dataset must contain a top-level list")
        return cls.from_records(content)

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "SWEbenchLoader":
        """逐行读取 JSONL；空行会被忽略，错误会包含准确行号。"""

        source = Path(path)
        records: list[Mapping[str, Any]] = []
        try:
            with source.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise DatasetFormatError(
                            f"invalid JSON on line {line_number} of {source}: {error}"
                        ) from error
                    if not isinstance(record, Mapping):
                        raise DatasetFormatError(
                            f"line {line_number} of {source} is not a JSON object"
                        )
                    records.append(record)
        except OSError as error:
            raise DatasetFormatError(
                f"cannot read JSONL dataset {source}: {error}"
            ) from error
        return cls.from_records(records)

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str = "princeton-nlp/SWE-bench_Verified",
        *,
        split: str = "test",
    ) -> "SWEbenchLoader":
        """通过可选的 Hugging Face 依赖下载数据集。

        该方法只属于环境准备阶段；正式离线求解期间禁止调用，避免网络访问
        破坏实验边界或引入随时间变化的数据。
        """

        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                'Hugging Face support requires: pip install -e ".[huggingface]"'
            ) from error

        dataset = load_dataset(dataset_name, split=split)
        return cls.from_records(dataset)

    def get(self, instance_id: str) -> SWEbenchTask:
        """按唯一 ID 返回任务；未知 ID 会转换成包含上下文的错误。"""

        try:
            return self._by_id[instance_id]
        except KeyError as error:
            raise KeyError(f"unknown SWE-bench instance_id: {instance_id}") from error

    def select(self, instance_ids: Iterable[str]) -> "SWEbenchLoader":
        """按调用方给出的顺序选择任务，用于固定实验题目顺序。"""

        return SWEbenchLoader(self.get(instance_id) for instance_id in instance_ids)

    def __len__(self) -> int:
        """返回安全任务总数，实现 ``Sequence`` 协议。"""

        return len(self._tasks)

    def __getitem__(self, index: int | slice) -> SWEbenchTask | tuple[SWEbenchTask, ...]:
        """支持单个索引和切片访问。"""

        return self._tasks[index]

    def __iter__(self) -> Iterator[SWEbenchTask]:
        """按数据集原始顺序迭代任务。"""

        return iter(self._tasks)
