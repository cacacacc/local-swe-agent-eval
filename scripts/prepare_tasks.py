"""Download SWE-bench Verified and export a fixed task snapshot without reference answers.

This script may only run with network access during environment preparation. It selects
records from the configured ID list, projects them through the ``SWEbenchTask`` whitelist,
and writes them to JSONL; the original ``patch`` and ``test_patch`` never enter the output
file, and the formal offline Agent only reads this safe snapshot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.swebench_loader import SWEbenchLoader
from experiment.config import ExperimentConfig


def _load_instance_ids(path: Path) -> list[str]:
    """Read the fixed ID array and reject empty lists, duplicates, and non-string values."""

    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read task ID list {path}: {error}") from error
    if not isinstance(value, list) or not value:
        raise ValueError(f"task ID list must be a non-empty JSON array: {path}")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError("every task ID must be a non-empty string")
    if len(set(value)) != len(value):
        raise ValueError("task ID list contains duplicates")
    return value


def prepare_snapshot(config: ExperimentConfig, destination: Path | str) -> Path:
    """Export tasks in configuration order and refuse to overwrite an existing snapshot to protect frozen evidence."""

    output = Path(destination).resolve()
    if output.exists():
        raise FileExistsError(f"task snapshot already exists: {output}")
    instance_ids = _load_instance_ids(config.tasks_path)
    loader = SWEbenchLoader.from_huggingface(
        config.dataset.name,
        split=config.dataset.split,
    ).select(instance_ids)

    output.parent.mkdir(parents=True, exist_ok=True)
    # The temporary file shares the target's directory so that replace stays atomic.
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for task in loader:
                handle.write(
                    json.dumps(
                        task.to_agent_payload(),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    """Declare the config input and the safe snapshot output location."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Load the strict config and generate the task snapshot."""

    arguments = build_parser().parse_args()
    config = ExperimentConfig.load(arguments.config)
    path = prepare_snapshot(config, arguments.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
