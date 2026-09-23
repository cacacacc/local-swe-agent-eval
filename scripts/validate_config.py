"""校验实验 YAML，并输出能够标识该实验的可复现性信息。"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment.config import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    """构建配置校验命令的参数解析器。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--require-frozen",
        action="store_true",
        help="Fail unless the config is explicitly frozen for formal evaluation.",
    )
    return parser


def main() -> int:
    """加载配置，可选检查冻结状态，并打印供人工核对的关键字段。"""

    arguments = build_parser().parse_args()
    config = ExperimentConfig.load(arguments.config)
    if arguments.require_frozen:
        config.require_frozen()

    # 使用稳定的 key=value 格式，既方便阅读，也便于 shell 脚本采集。
    print(f"config={config.source_path}")
    print(f"fingerprint={config.fingerprint}")
    print(f"phase={config.experiment.phase}")
    print(f"frozen={str(config.experiment.configuration_frozen).lower()}")
    print(f"model={config.model.name}")
    print(f"prompt={config.prompt_template_path}")
    print(f"tasks={config.tasks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
