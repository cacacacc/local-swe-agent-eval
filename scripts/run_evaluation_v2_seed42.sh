#!/usr/bin/env bash

# 一键启动 evaluation v2 的 seed-42 固定题集。
#
# 用法：
#   ./scripts/run_evaluation_v2_seed42.sh 15
#   ./scripts/run_evaluation_v2_seed42.sh 20
#   ./scripts/run_evaluation_v2_seed42.sh 30
#
# 入口会把项目的 .venv 放到 PATH 最前面，因此调用者无需提前激活虚拟环境。
# 正式运行前会按需启动 Docker Desktop 和 Ollama，并等待服务真正可用；脚本
# 不会安装软件，也不会修改 Docker Desktop 的 WSL Integration 等主机设置。
#
# 可通过环境变量覆盖本机相关路径或运行标识：
#   BATCH_ID                批次 ID；默认包含所选题数
#   SWEBENCH_ROOT           SWE-bench 仓库；默认 $HOME/src/SWE-bench
#   LOCAL_MODEL_BASE_URL    Ollama 地址；默认 http://localhost:11434

set -Eeuo pipefail

# 只接受仓库中已经冻结并验证的题集规模，防止拼写错误指向不存在的配置。
TASK_COUNT="${1:-20}"
case "${TASK_COUNT}" in
    15|20|30) ;;
    *)
        printf '用法：%s {15|20|30}\n' "$0" >&2
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

ollama_ready() {
    # 使用项目 Python 探测 API，避免要求宿主机额外安装 curl。
    "${PYTHON_BIN}" -c '
import sys
import urllib.request

base_url = sys.argv[1].rstrip("/")
with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
    if response.status != 200:
        raise SystemExit(1)
' "${LOCAL_MODEL_BASE_URL}" >/dev/null 2>&1
}

wait_for_docker() {
    # Docker Desktop 启动通常需要数十秒；同时检查命令与 daemon，才能覆盖
    # WSL Integration 尚未挂载客户端以及 daemon 尚未就绪两种状态。
    local attempt
    for ((attempt = 1; attempt <= 45; attempt++)); do
        if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}

wait_for_ollama() {
    local attempt
    for ((attempt = 1; attempt <= 30; attempt++)); do
        if ollama_ready; then
            return 0
        fi
        sleep 2
    done
    return 1
}

start_docker_if_needed() {
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        return
    fi

    printf 'Docker 尚未就绪，正在启动 Docker Desktop...\n'
    if command -v powershell.exe >/dev/null 2>&1; then
        # WSL 无法直接管理 Windows 服务；通过 Windows 启动已安装的 Desktop，
        # 随后仍以 docker info 为准等待，而不是假定进程出现即代表可用。
        powershell.exe -NoProfile -NonInteractive -Command \
            'Start-Process "$Env:ProgramFiles\Docker\Docker\Docker Desktop.exe"' \
            >/dev/null 2>&1 || true
    fi

    if ! wait_for_docker; then
        cat >&2 <<'EOF'
启动检查失败：Docker Desktop 启动后仍无法访问 Docker daemon。

请确认 Windows 已安装 Docker Desktop，并在以下位置启用当前发行版：
  Settings -> Resources -> WSL Integration
EOF
        exit 1
    fi
    printf 'Docker 已就绪。\n'
}

start_ollama_if_needed() {
    if ollama_ready; then
        return
    fi

    printf 'Ollama 尚未就绪，正在启动服务...\n'
    if command -v ollama >/dev/null 2>&1; then
        # 日志放在 /tmp，避免把宿主服务日志写进实验仓库或提交到 Git。
        nohup ollama serve >/tmp/local-swe-agent-eval-ollama.log 2>&1 &
    elif command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -NonInteractive -Command \
            'Start-Process -WindowStyle Hidden ollama -ArgumentList "serve"' \
            >/dev/null 2>&1 || true
    else
        fail "找不到 ollama 或 powershell.exe，无法自动启动 Ollama。"
    fi

    if ! wait_for_ollama; then
        fail "自动启动 Ollama 后仍无法访问 ${LOCAL_MODEL_BASE_URL}；请检查 Ollama 安装和服务日志。"
    fi
    printf 'Ollama 已就绪。\n'
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
    fail "找不到项目虚拟环境 ${PYTHON_BIN}，请先创建并安装项目依赖。"
fi

# Bash 脚本不能修改调用它的父 shell，但把 .venv/bin 放到当前脚本 PATH 的
# 最前面与激活环境对本次评测的效果一致，子进程调用 `python` 时也不会失效。
export PATH="${PROJECT_ROOT}/.venv/bin:${PATH}"

if [[ ! -f "${CONFIG_PATH}" || ! -f "${TASKS_PATH}" ]]; then
    fail "${TASK_COUNT} 题配置或准备后的任务文件不存在，请先完成数据准备。"
fi

start_docker_if_needed

if ! command -v claude >/dev/null 2>&1; then
    fail "找不到 claude 命令；请先安装 Claude Code，并确认它位于 PATH。"
fi

if [[ ! -x "${SWEBENCH_ROOT}/.venv/bin/swebench" ]]; then
    fail "找不到 ${SWEBENCH_ROOT}/.venv/bin/swebench，请检查 SWEBENCH_ROOT 或安装 SWE-bench。"
fi

start_ollama_if_needed

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
