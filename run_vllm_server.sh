#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

source lib.sh

check_deps
check_nvidia

FORCE_MODEL_SELECT=false
REBUILD_FLAG=""
MODE="run"
WAIT_TIMEOUT=300
CLI_PORT=""
CLI_CONTEXT=""
CLI_GPU_UTIL=""
CLI_DTYPE=""
CLI_TP=""

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model, -m           Re-select model interactively"
    echo "  --port, -p PORT       Server port (default: from .env; enables multi-instance)"
    echo "  --context, -c SIZE    Set max model length (tokens)"
    echo "  --gpu-util FLOAT      GPU memory utilization 0.0-1.0 (default: 0.95)"
    echo "  --dtype DTYPE         Data type: auto, float16, bfloat16, float8 (default: auto)"
    echo "  --tp N                Tensor parallel size (default: 1)"
    echo "  --rebuild             Force Docker image rebuild"
    echo "  --status              Show server status (all instances if no --port)"
    echo "  --wait SECONDS        Max wait for health (default: ${WAIT_TIMEOUT}, 0=skip)"
    echo "  --help, -h            Show this help"
    echo ""
    echo "Multi-instance: each port gets its own config (.env.<port>) and container."
    echo "  $0                       # default instance"
    echo "  $0 -p 8081 -m            # second instance on 8081"
    echo ""
    echo "Extra server args: edit vllm.args (one per line, shared across instances)"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)         FORCE_MODEL_SELECT=true; shift ;;
        --port|-p)          CLI_PORT="${2:?--port requires a value}"; shift 2 ;;
        --context|-c)       CLI_CONTEXT="${2:?--context requires a value}"; shift 2 ;;
        --gpu-util)         CLI_GPU_UTIL="${2:?--gpu-util requires a value}"; shift 2 ;;
        --dtype)            CLI_DTYPE="${2:?--dtype requires a value}"; shift 2 ;;
        --tp)               CLI_TP="${2:?--tp requires a value}"; shift 2 ;;
        --rebuild)          REBUILD_FLAG="--build"; shift ;;
        --status)           MODE="status"; shift ;;
        --wait)             WAIT_TIMEOUT="${2:?--wait requires a value}"; shift 2 ;;
        --help|-h)          show_help; exit 0 ;;
        *)                  echo "Unknown option: $1. Use --help for usage."; exit 1 ;;
    esac
done

BASE_PORT="8080"
if [ -f .env ]; then
    BP=$(grep -E "^HOST_PORT=" .env 2>/dev/null | tail -1 | cut -d= -f2-)
    [ -n "$BP" ] && BASE_PORT="$BP"
fi

PORT="${CLI_PORT:-$BASE_PORT}"

if [ "$PORT" = "$BASE_PORT" ]; then
    INSTANCE_ENV=".env"
else
    INSTANCE_ENV=".env.${PORT}"
fi

PROJECT_NAME="vllm-${PORT}"

dc() {
    docker compose -p "$PROJECT_NAME" --env-file "$INSTANCE_ENV" "$@"
}

source_shared_env() {
    [ -f gpu.env ] || { [ -f gpu.env.example ] && cp gpu.env.example gpu.env; }
    [ -f vllm.args ] || { [ -f vllm.args.example ] && cp vllm.args.example vllm.args; }
    set -a
    [ -f gpu.env ] && source gpu.env
    set +a
}

source_instance_env() {
    if [ ! -f "$INSTANCE_ENV" ]; then
        if [ -f .env ]; then
            cp .env "$INSTANCE_ENV"
        elif [ -f .env.example ]; then
            cp .env.example "$INSTANCE_ENV"
        else
            log_error "No .env or .env.example found"
            exit 1
        fi
    fi
    set -a
    source "$INSTANCE_ENV"
    set +a
}

source_shared_env
source_instance_env

if [ -n "$CLI_PORT" ]; then
    set_env_var "HOST_PORT" "$CLI_PORT" "$INSTANCE_ENV"
    set_env_var "PORT" "$CLI_PORT" "$INSTANCE_ENV"
    HOST_PORT="$CLI_PORT"
fi

if [ -n "$CLI_CONTEXT" ]; then
    set_env_var "MAX_MODEL_LEN" "$CLI_CONTEXT" "$INSTANCE_ENV"
    MAX_MODEL_LEN="$CLI_CONTEXT"
fi

if [ -n "$CLI_GPU_UTIL" ]; then
    set_env_var "GPU_MEMORY_UTILIZATION" "$CLI_GPU_UTIL" "$INSTANCE_ENV"
    GPU_MEMORY_UTILIZATION="$CLI_GPU_UTIL"
fi

if [ -n "$CLI_DTYPE" ]; then
    set_env_var "DTYPE" "$CLI_DTYPE" "$INSTANCE_ENV"
    DTYPE="$CLI_DTYPE"
fi

if [ -n "$CLI_TP" ]; then
    set_env_var "TENSOR_PARALLEL_SIZE" "$CLI_TP" "$INSTANCE_ENV"
    TENSOR_PARALLEL_SIZE="$CLI_TP"
fi

ACTIVE_VERSION=$(detect_active_version)

show_instance_status() {
    local port="$1"
    local env_file="$2"
    local project="vllm-${port}"

    local model="not set" ctx="8192" gpu_util="0.95"
    if [ -f "$env_file" ]; then
        model=$(grep -E "^MODEL_NAME=" "$env_file" 2>/dev/null | tail -1 | cut -d= -f2- || true)
        model="${model:-not set}"
        ctx=$(grep -E "^MAX_MODEL_LEN=" "$env_file" 2>/dev/null | tail -1 | cut -d= -f2- || true)
        ctx="${ctx:-8192}"
        gpu_util=$(grep -E "^GPU_MEMORY_UTILIZATION=" "$env_file" 2>/dev/null | tail -1 | cut -d= -f2- || true)
        gpu_util="${gpu_util:-0.95}"
    fi

    echo "  Port:     ${port}"
    echo "  Version:  ${ACTIVE_VERSION}"
    echo "  Model:    ${model}"
    echo "  Max len:  ${ctx}"
    echo "  GPU util: ${gpu_util}"

    local container_id
    container_id=$(docker compose -p "$project" ps -q vllm-server 2>/dev/null | head -1 || true)
    if [ -n "$container_id" ]; then
        echo "  State:    running"
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            echo "  Health:   OK"
        else
            echo "  Health:   not ready"
        fi
    else
        echo "  State:    stopped"
    fi
}

if [ "$MODE" = "status" ]; then
    if [ -n "$CLI_PORT" ]; then
        echo "=== vllm-server :${PORT} ==="
        show_instance_status "$PORT" "$INSTANCE_ENV"
    else
        echo "=== vllm-server instances ==="
        echo ""
        echo "  [:${BASE_PORT}]"
        show_instance_status "$BASE_PORT" ".env"
        for env_file in .env.*; do
            [ -f "$env_file" ] || continue
            local_port="${env_file#.env.}"
            [[ "$local_port" =~ ^[0-9]+$ ]] || continue
            echo ""
            echo "  [:${local_port}]"
            show_instance_status "$local_port" "$env_file"
        done
    fi
    exit 0
fi

if [ -z "${MODEL_NAME:-}" ] || [ "$FORCE_MODEL_SELECT" = true ]; then
    SELECTED_MODEL=$(./select_model.sh | head -1)
    if [ -z "$SELECTED_MODEL" ]; then
        echo "No model selected. Exiting."
        exit 1
    fi

    set_model_in_env "$SELECTED_MODEL" "$INSTANCE_ENV"
    set -a; source "$INSTANCE_ENV"; set +a
fi

if [ -n "$REBUILD_FLAG" ]; then
    export VLLM_VERSION="$ACTIVE_VERSION"
    docker compose build vllm-server 2>&1 | tail -5
fi

if dc ps -q vllm-server 2>/dev/null | grep -q .; then
    dc down 2>/dev/null
fi

echo "vllm-server starting  version=${ACTIVE_VERSION}  model=${MODEL_NAME}  max_len=${MAX_MODEL_LEN:-8192}  gpu_util=${GPU_MEMORY_UTILIZATION:-0.95}  port=${PORT}"

dc up -d

if [ "$WAIT_TIMEOUT" -gt 0 ] && command -v curl &>/dev/null; then
    echo -n "Waiting for server to become healthy"
    ELAPSED=0
    while [ "$ELAPSED" -lt "$WAIT_TIMEOUT" ]; do
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            echo ""
            echo "Server ready at http://localhost:${PORT}"
            echo "  logs: docker compose -p ${PROJECT_NAME} logs -f vllm-server  |  stop: ./stop_vllm_server.sh ${PORT}"
            exit 0
        fi
        echo -n "."
        sleep 5
        ELAPSED=$((ELAPSED + 5))
    done
    echo ""
    echo "Server not ready after ${WAIT_TIMEOUT}s — check logs: docker compose -p ${PROJECT_NAME} logs vllm-server"
    exit 1
else
    echo "http://localhost:${PORT}  |  stop: ./stop_vllm_server.sh ${PORT}"
fi
