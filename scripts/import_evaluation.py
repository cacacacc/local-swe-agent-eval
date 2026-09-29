"""Safely associate the official SWE-bench ``results.json`` with a local run."""

from __future__ import annotations

import argparse
from pathlib import Path

from tracking.evaluation_result import import_official_evaluation


def build_parser() -> argparse.ArgumentParser:
    """Declare explicit path and harness identity parameters, avoiding unreliable inference from directory names."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-path", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--harness-run-id", required=True)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    return parser


def main() -> int:
    """Import the official verdict and print the updated result file path."""

    arguments = build_parser().parse_args()
    destination = import_official_evaluation(
        arguments.run_path,
        arguments.report,
        harness_run_id=arguments.harness_run_id,
        dataset=arguments.dataset,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
