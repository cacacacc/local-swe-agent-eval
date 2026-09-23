"""Load SWE-bench tasks without exposing reference solutions to the agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .task import SWEbenchTask, TaskValidationError


class DatasetFormatError(ValueError):
    """Raised when a task collection is malformed or internally inconsistent."""


class SWEbenchLoader(Sequence[SWEbenchTask]):
    """An ordered, duplicate-free collection of safe SWE-bench tasks."""

    def __init__(self, tasks: Iterable[SWEbenchTask]) -> None:
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
        """Download a dataset through the optional Hugging Face dependency.

        This method belongs to environment preparation and must not be called
        during a formal offline solving run.
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
        try:
            return self._by_id[instance_id]
        except KeyError as error:
            raise KeyError(f"unknown SWE-bench instance_id: {instance_id}") from error

    def select(self, instance_ids: Iterable[str]) -> "SWEbenchLoader":
        """Select tasks in the caller-provided order."""

        return SWEbenchLoader(self.get(instance_id) for instance_id in instance_ids)

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, index: int | slice) -> SWEbenchTask | tuple[SWEbenchTask, ...]:
        return self._tasks[index]

    def __iter__(self) -> Iterator[SWEbenchTask]:
        return iter(self._tasks)
