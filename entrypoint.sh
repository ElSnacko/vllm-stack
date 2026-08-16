#!/bin/bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-}"
MODEL_DIR="/app/models/${MODEL_NAME}"

if [ -n "${MODEL_NAME}" ] && [ ! -d "${MODEL_DIR}" ]; then
    echo "Error: Model directory not found: ${MODEL_DIR}" >&2
    echo "Check MODEL_NAME in .env — currently set to: ${MODEL_NAME}" >&2
    exit 1
fi

ARGS=("--model" "${MODEL_DIR}")

ARGS+=("--host" "${HOST}" "--port" "${PORT}")

MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
ARGS+=("--max-model-len" "${MAX_MODEL_LEN}")

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
ARGS+=("--gpu-memory-utilization" "${GPU_MEMORY_UTILIZATION}")

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
if [ "$TENSOR_PARALLEL_SIZE" -gt 1 ]; then
    ARGS+=("--tensor-parallel-size" "${TENSOR_PARALLEL_SIZE}")
fi

DTYPE="${DTYPE:-auto}"
if [ "$DTYPE" != "auto" ]; then
    ARGS+=("--dtype" "${DTYPE}")
fi

TASK="${TASK:-}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-}"
if [ -z "$TASK" ] && [ -f "${MODEL_DIR}/config.json" ]; then
    if ! [ -f "${MODEL_DIR}/preprocessor_config.json" ] && grep -q "ConditionalGeneration" "${MODEL_DIR}/config.json" 2>/dev/null; then
        TASK="generate"
        LANGUAGE_MODEL_ONLY="1"
    fi
fi
if [ -n "$TASK" ]; then
    ARGS+=("--runner" "${TASK}")
fi
if [ "$LANGUAGE_MODEL_ONLY" = "1" ]; then
    ARGS+=("--language-model-only")
fi

TRUST_CHAT_TEMPLATE="${TRUST_CHAT_TEMPLATE:-}"
if [ "$TRUST_CHAT_TEMPLATE" = "1" ]; then
    ARGS+=("--trust-request-chat-template")
fi

CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
if [ -n "$CHAT_TEMPLATE" ]; then
    ARGS+=("--chat-template" "$CHAT_TEMPLATE")
fi

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
if [ "$KV_CACHE_DTYPE" != "auto" ]; then
    ARGS+=("--kv-cache-dtype" "${KV_CACHE_DTYPE}")
fi

COMPILATION_CONFIG="${COMPILATION_CONFIG:-}"
if [ -n "$COMPILATION_CONFIG" ]; then
    ARGS+=("--compilation-config" "$COMPILATION_CONFIG")
fi

SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"
if [ -n "$SPECULATIVE_CONFIG" ]; then
    ARGS+=("--speculative-config" "$SPECULATIVE_CONFIG")
fi

ENFORCE_EAGER="${ENFORCE_EAGER:-}"
if [ "$ENFORCE_EAGER" = "1" ]; then
    ARGS+=("--enforce-eager")
fi

ARGS_FILE_ARGS=""
if [ -f "/app/vllm.args" ]; then
    ARGS_FILE_ARGS=$(sed '/^#/d; /^[[:space:]]*$/d; s/[[:space:]]*$//' /app/vllm.args)
fi

while IFS= read -r line; do
    [ -z "$line" ] && continue
    eval 'ARGS+=('$line')'
done <<< "${ARGS_FILE_ARGS}"

EXTRA_ARGS="${EXTRA_ARGS:-}"
if [ -n "$EXTRA_ARGS" ]; then
    eval 'ARGS+=('$EXTRA_ARGS')'
fi

echo "========================================"
echo " vLLM OpenAI-compatible server"
echo "========================================"
echo " Model:       ${MODEL_NAME}"
echo " Max tokens:  ${MAX_MODEL_LEN}"
echo " GPU util:    ${GPU_MEMORY_UTILIZATION}"
echo " Tensor par:  ${TENSOR_PARALLEL_SIZE}"
echo " Dtype:       ${DTYPE}"
echo " Host:        ${HOST}:${PORT}"
echo "========================================"

exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
