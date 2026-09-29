"""Validate the experiment YAML and print reproducibility information identifying the experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment.config import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the config validation command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--require-frozen",
        action="store_true",
        help="Fail unless the config is explicitly frozen for formal evaluation.",
    )
    return parser


def main() -> int:
    """Load the config, optionally check frozen status, and print key fields for manual review."""

    arguments = build_parser().parse_args()
    config = ExperimentConfig.load(arguments.config)
    if arguments.require_frozen:
        config.require_frozen()

    # Use a stable key=value format, which is easy to read and convenient for shell scripts to capture.
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
