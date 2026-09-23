"""下载 SWE-bench Verified，并导出不含参考答案的固定任务快照。

本脚本只能在环境准备阶段联网运行。它从配置中的 ID 清单选择记录，再经
``SWEbenchTask`` 白名单投影后写入 JSONL；原始 ``patch`` 与 ``test_patch`` 从不进入
输出文件，正式离线 Agent 只读取该安全快照。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.swebench_loader import SWEbenchLoader
from experiment.config import ExperimentConfig


def _load_instance_ids(path: Path) -> list[str]:
    """读取固定 ID 数组，并拒绝空列表、重复项和非字符串值。"""

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
    """按配置顺序导出任务，并拒绝覆盖已有快照以保护冻结证据。"""

    output = Path(destination).resolve()
    if output.exists():
        raise FileExistsError(f"task snapshot already exists: {output}")
    instance_ids = _load_instance_ids(config.tasks_path)
    loader = SWEbenchLoader.from_huggingface(
        config.dataset.name,
        split=config.dataset.split,
    ).select(instance_ids)

    output.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件与目标位于同一目录，replace 才能保持原子性。
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
    """声明配置输入和安全快照输出位置。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """加载严格配置并生成任务快照。"""

    arguments = build_parser().parse_args()
    config = ExperimentConfig.load(arguments.config)
    path = prepare_snapshot(config, arguments.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
