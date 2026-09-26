#!/usr/bin/env bash

# 一键启动 evaluation v2 的 seed-42 固定题集。
#
# 用法：
#   ./scripts/run_evaluation_v2_seed42.sh 15
#   ./scripts/run_evaluation_v2_seed42.sh 20
#
# 入口直接使用项目的 .venv Python，因此调用者无需提前激活虚拟环境。
# 正式运行前会检查本机依赖，确保 Docker、Ollama 或 SWE-bench 问题不会在
# 创建批次产物后才暴露。脚本只负责检查，不会安装软件或修改主机设置。
#
# 可通过环境变量覆盖本机相关路径或运行标识：
#   BATCH_ID                批次 ID；默认包含所选题数
#   SWEBENCH_ROOT           SWE-bench 仓库；默认 $HOME/src/SWE-bench
#   LOCAL_MODEL_BASE_URL    Ollama 地址；默认 http://localhost:11434

set -Eeuo pipefail

# 只接受仓库中已经冻结并验证的题集规模，防止拼写错误指向不存在的配置。
TASK_COUNT="${1:-20}"
case "${TASK_COUNT}" in
    15|20) ;;
    *)
        printf '用法：%s {15|20}\n' "$0" >&2
        exit 2
        ;;
esac

# 以脚本自身位置定位项目，避免要求用户从仓库根目录执行命令。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
CONFIG_PATH="${PROJECT_ROOT}/configs/evaluation_v2_${TASK_COUNT}_seed42.yaml"
TASKS_PATH="${PROJECT_ROOT}/prepared/evaluation_tasks_${TASK_COUNT}_seed42.jsonl"
SWEBENCH_ROOT="${SWEBENCH_ROOT:-${HOME}/src/SWE-bench}"
BATCH_ID="${BATCH_ID:-evaluation-v2-qwen35-${TASK_COUNT}-seed42}"
LOCAL_MODEL_BASE_URL="${LOCAL_MODEL_BASE_URL:-http://localhost:11434}"

fail() {
    # 所有预检错误采用统一格式，并保证失败时不会继续创建实验目录。
    printf '启动检查失败：%s\n' "$1" >&2
    exit 1
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
    fail "找不到项目虚拟环境 ${PYTHON_BIN}，请先创建并安装项目依赖。"
fi

if [[ ! -f "${CONFIG_PATH}" || ! -f "${TASKS_PATH}" ]]; then
    fail "${TASK_COUNT} 题配置或准备后的任务文件不存在，请先完成数据准备。"
fi

if ! command -v docker >/dev/null 2>&1; then
    cat >&2 <<'EOF'
启动检查失败：当前 WSL 中找不到 docker 命令。

请在 Windows 的 Docker Desktop 中打开：
  Settings -> Resources -> WSL Integration
然后启用当前发行版，点击 Apply & Restart。
回到 WSL 后先运行 `docker version` 确认客户端和服务端均可访问。
EOF
    exit 1
fi

# `docker` 命令存在并不代表 daemon 可用；Docker Desktop 未启动或 WSL
# integration 尚未生效时，必须在下载镜像之前给出清晰错误。
if ! docker info >/dev/null 2>&1; then
    fail "Docker daemon 不可访问；请启动 Docker Desktop，并确认 WSL Integration 已启用。"
fi

if ! command -v claude >/dev/null 2>&1; then
    fail "找不到 claude 命令；请先安装 Claude Code，并确认它位于 PATH。"
fi

if [[ ! -x "${SWEBENCH_ROOT}/.venv/bin/swebench" ]]; then
    fail "找不到 ${SWEBENCH_ROOT}/.venv/bin/swebench，请检查 SWEBENCH_ROOT 或安装 SWE-bench。"
fi

# 使用 Python 标准库探测 Ollama，避免额外依赖 curl。这里只访问 loopback
# 服务，不进行外网请求，也不负责替用户启动或修改模型服务。
if ! "${PYTHON_BIN}" -c '
import sys
import urllib.request

base_url = sys.argv[1].rstrip("/")
with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
    if response.status != 200:
        raise SystemExit(1)
' "${LOCAL_MODEL_BASE_URL}" >/dev/null 2>&1; then
    fail "无法访问 ${LOCAL_MODEL_BASE_URL}；请先启动 Ollama，并确认 qwen3.5:9b 已安装。"
fi

printf '启动检查通过，开始运行 %s 题批次：%s\n' "${TASK_COUNT}" "${BATCH_ID}"

# Python 的模块入口依赖仓库根目录位于 import path；这里显式切换目录，确保
# 用户从任意工作目录调用本脚本时行为一致。
cd -- "${PROJECT_ROOT}"

# exec 让批处理直接接管当前终端，Ctrl-C 和退出码可准确传递给调用者。
exec "${PYTHON_BIN}" -m scripts.run_batch \
    --config "${CONFIG_PATH}" \
    --tasks "${TASKS_PATH}" \
    --batch-id "${BATCH_ID}" \
    --swebench-root "${SWEBENCH_ROOT}" \
    --base-url "${LOCAL_MODEL_BASE_URL}" \
    --expected-tasks "${TASK_COUNT}" \
    --allow-network-preparation
