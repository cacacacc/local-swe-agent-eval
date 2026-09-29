"""Load SWE-bench tasks while keeping reference answers out of the Agent context.

Supports local JSON, JSONL, and an optional Hugging Face data source. Regardless of
where the input comes from, it must ultimately pass the field whitelist and format
validation of :class:`SWEbenchTask`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .task import SWEbenchTask, TaskValidationError


class DatasetFormatError(ValueError):
    """Raised when the task collection is malformed or internally inconsistent."""


class SWEbenchLoader(Sequence[SWEbenchTask]):
    """A safe task collection that preserves input order and contains no duplicate instance IDs."""

    def __init__(self, tasks: Iterable[SWEbenchTask]) -> None:
        # The tuple keeps order stable after loading, while the dict provides O(1) lookup by ID.
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
        """Convert a batch of raw records into a safely validated task collection."""

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
        """Load tasks from a UTF-8 JSON file whose top level is an array."""

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
        """Read JSONL line by line; blank lines are ignored and errors carry the exact line number."""

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
        """Download the dataset via the optional Hugging Face dependency.

        This method belongs only to the environment-preparation phase; it must not be
        called during formal offline solving, to prevent network access from breaking
        the experiment boundary or introducing time-varying data.
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
        """Return the task by its unique ID; an unknown ID is turned into an error with context."""

        try:
            return self._by_id[instance_id]
        except KeyError as error:
            raise KeyError(f"unknown SWE-bench instance_id: {instance_id}") from error

    def select(self, instance_ids: Iterable[str]) -> "SWEbenchLoader":
        """Select tasks in the caller's given order, used to pin the experiment's task order."""

        return SWEbenchLoader(self.get(instance_id) for instance_id in instance_ids)

    def __len__(self) -> int:
        """Return the total number of safe tasks, implementing the ``Sequence`` protocol."""

        return len(self._tasks)

    def __getitem__(self, index: int | slice) -> SWEbenchTask | tuple[SWEbenchTask, ...]:
        """Support single-index and slice access."""

        return self._tasks[index]

    def __iter__(self) -> Iterator[SWEbenchTask]:
        """Iterate over tasks in the dataset's original order."""

        return iter(self._tasks)
