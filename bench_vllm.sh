#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

source lib.sh

PROMPT_TOKENS="512,1024,2048"
GEN_TOKENS="128,256"
BENCH_REPS=3
OUTPUT_FILE=""
FORCE_MODEL=false
API_BASE=""

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Benchmark a running vLLM server using the OpenAI-compatible API.

The server must already be running (./run_vllm_server.sh).

Options:
  -p, --prompt SIZES      Prompt sizes to test (default: ${PROMPT_TOKENS})
  -n, --tokens SIZES      Token counts to generate (default: ${GEN_TOKENS})
  -r, --reps N            Repetitions per test (default: ${BENCH_REPS})
  -u, --url URL           API base URL (default: http://localhost:8080)
  -o, --output FILE       Save results to file
  -m, --model             Re-select model interactively
  -h, --help              Show this help

Examples:
  $0                                    Default benchmark
  $0 -p 512,1024 -n 128,256            Custom prompt/token sizes
  $0 -u http://localhost:8081           Benchmark specific instance
  $0 -o results.txt                     Save results
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--prompt)    PROMPT_TOKENS="$2"; shift 2 ;;
        -n|--tokens)    GEN_TOKENS="$2"; shift 2 ;;
        -r|--reps)      BENCH_REPS="$2"; shift 2 ;;
        -u|--url)       API_BASE="$2"; shift 2 ;;
        -o|--output)    OUTPUT_FILE="$2"; shift 2 ;;
        -m|--model)     FORCE_MODEL=true; shift ;;
        -h|--help)      show_help; exit 0 ;;
        *)              echo "Unknown option: $1. Use --help for usage."; exit 1 ;;
    esac
done

source_env_files

API_BASE="${API_BASE:-http://localhost:${HOST_PORT:-8080}}"

ACTIVE_VERSION=$(detect_active_version)

if [ "$FORCE_MODEL" = true ]; then
    SELECTED_MODEL=$(./select_model.sh | head -1)
    if [ -z "$SELECTED_MODEL" ]; then
        echo "No model selected. Exiting."
        exit 1
    fi
    set_model_in_env "$SELECTED_MODEL"
    source .env
fi

MODEL_NAME="${MODEL_NAME:-not set}"

if ! curl -sf "${API_BASE}/health" >/dev/null 2>&1; then
    echo "Error: vLLM server not responding at ${API_BASE}"
    echo "Start it with: ./run_vllm_server.sh"
    exit 1
fi

MODEL_ID=$(curl -sf "${API_BASE}/v1/models" 2>/dev/null | jq -r '.data[0].id' 2>/dev/null || echo "$MODEL_NAME")

echo ""
echo "=== vLLM Benchmark ==="
echo "  Version:    ${ACTIVE_VERSION}"
echo "  Model:      ${MODEL_ID}"
echo "  API:        ${API_BASE}"
echo "  PP sizes:   ${PROMPT_TOKENS}"
echo "  TG sizes:   ${GEN_TOKENS}"
echo "  Reps:       ${BENCH_REPS}"
echo "======================="
echo ""

TMP_RESULTS=$(mktemp /tmp/vllm-bench-XXXXXX.txt)

bench_prompt() {
    local num_tokens="$1"
    local prompt=$(python3 -c "print('x' * $num_tokens)" 2>/dev/null || echo "")
    [ -z "$prompt" ] && return

    echo -n "  PP ${num_tokens}: "

    local total_time=0
    for ((rep=1; rep<=BENCH_REPS; rep++)); do
        local start_time
        start_time=$(date +%s%N)

        curl -sf "${API_BASE}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"${MODEL_ID}\", \"prompt\": \"${prompt}\", \"max_tokens\": 1, \"temperature\": 0}" \
            >/dev/null 2>&1 || { echo "FAILED"; return 1; }

        local end_time
        end_time=$(date +%s%N)
        local elapsed=$(( (end_time - start_time) / 1000000 ))
        total_time=$((total_time + elapsed))
    done

    local avg_ms=$((total_time / BENCH_REPS))
    local tps
    tps=$(awk -v t="$num_tokens" -v ms="$avg_ms" 'BEGIN {printf "%.1f", t / (ms/1000)}')
    echo "${tps} t/s  (${avg_ms} ms avg)"
    echo "PP ${num_tokens}|${tps}|${avg_ms}" >> "$TMP_RESULTS"
}

bench_generate() {
    local num_tokens="$1"
    local prompt="Write a story."

    echo -n "  TG ${num_tokens}: "

    local total_ms=0
    local total_tokens=0
    for ((rep=1; rep<=BENCH_REPS; rep++)); do
        local start_ns end_ns elapsed_ms
        start_ns=$(date +%s%N)

        local response
        response=$(curl -sf "${API_BASE}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"${MODEL_ID}\", \"prompt\": \"${prompt}\", \"max_tokens\": ${num_tokens}, \"temperature\": 0, \"stream\": false}" \
            2>/dev/null || echo "{}")

        end_ns=$(date +%s%N)
        elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

        local tokens
        tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0' 2>/dev/null || echo 0)
        total_ms=$((total_ms + elapsed_ms))
        total_tokens=$((total_tokens + tokens))
    done

    if [ "$total_tokens" -gt 0 ] 2>/dev/null; then
        local tps
        tps=$(awk -v t="$total_tokens" -v ms="$total_ms" 'BEGIN {printf "%.1f", t / (ms/1000)}')
        echo "${tps} t/s  (${total_tokens} tokens in ${total_ms} ms, ${BENCH_REPS} reps)"
        echo "TG ${num_tokens}|${tps}|${total_ms}" >> "$TMP_RESULTS"
    else
        echo "no data"
    fi
}

echo "--- Prompt Processing (PP) ---"
IFS=',' read -ra PP_SIZES <<< "$PROMPT_TOKENS"
for size in "${PP_SIZES[@]}"; do
    bench_prompt "$size"
done

echo ""
echo "--- Token Generation (TG) ---"
IFS=',' read -ra TG_SIZES <<< "$GEN_TOKENS"
for size in "${TG_SIZES[@]}"; do
    bench_generate "$size"
done

echo ""
echo "======================="

if [ -n "$OUTPUT_FILE" ] && [ -f "$TMP_RESULTS" ]; then
    cp "$TMP_RESULTS" "$OUTPUT_FILE"
    echo "Results saved to: $OUTPUT_FILE"
fi

rm -f "$TMP_RESULTS"
echo "Version: ${ACTIVE_VERSION}"
