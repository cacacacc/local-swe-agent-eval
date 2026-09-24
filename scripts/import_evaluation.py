"""把 SWE-bench 官方 ``results.json`` 安全关联到一个本地 run。"""

from __future__ import annotations

import argparse
from pathlib import Path

from tracking.evaluation_result import import_official_evaluation


def build_parser() -> argparse.ArgumentParser:
    """声明显式路径和 harness 身份参数，避免根据目录名作不可靠推断。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-path", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--harness-run-id", required=True)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    return parser


def main() -> int:
    """导入官方判定并输出被更新的 result 文件路径。"""

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
