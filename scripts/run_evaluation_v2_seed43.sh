#!/usr/bin/env bash

# 启动与所有既有题集零重叠的 seed-43 固定 30 题。
#
# 环境准备、断点续跑和正式批处理逻辑全部复用经过测试的 seed42 入口；这里只
# 固定 TASK_SEED，避免复制长脚本后两份启动边界逐渐分叉。

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TASK_SEED=43

exec "${SCRIPT_DIR}/run_evaluation_v2_seed42.sh" 30
