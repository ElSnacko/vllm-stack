#!/bin/bash
# KVarN vs native KV cache — side-by-side benchmark on Qwen3-8B
#
# Comparison:
#   Baseline: native KV cache (fp16/bf16 auto)
#   KVarN:    same model, kv_cache_dtype=kvarn_k4v2_g128 (4-bit keys, 2-bit values)
#
# KVarN requires head_dim=128. Qwen3.6-27B has head_dim=256 (hybrid Mamba arch)
# and is blocked by issue #10 (fix in progress by maintainer). Using Qwen3-8B-FP8
# instead — standard transformer, head_dim=128, fits easily in 32GB.
#
# Usage:
#   ./test_kvarn.sh                    full suite
#   ./test_kvarn.sh --accuracy-only
#   ./test_kvarn.sh --throughput-only

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen3-8B-FP8}"
MODEL_DIR="${MODEL_DIR:-/mnt/bignvme/ai-stack/models/huggingface}"

# KV_CACHE_DTYPE for each run
BASELINE_KV_DTYPE="${BASELINE_KV_DTYPE:-auto}"
KVARN_KV_DTYPE="kvarn_k4v2_g128"

PORT_BASE=8181
PORT_KVARN=8182

GPU_MEM="${GPU_MEM:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"   # 32K — meaningful KV pressure for 8B
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 2 4 8}"
NUM_OUTPUT_TOKENS=256

# Standard flags — Qwen3-8B is a plain causal LM, no Mamba or VL overhead.
SHARED_EXTRA_ARGS="--enable-prefix-caching --max-cudagraph-capture-size 128"

BASELINE_IMAGE="vllm-baseline:test"
KVARN_IMAGE="vllm-kvarn:test"

RESULTS_DIR="$(pwd)/kvarn_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

log()  { echo "[$(date +%H:%M:%S)] $*" >&2; }
fail() { echo "ERROR: $*" >&2; exit 1; }
# All output from start_server is captured into CID — only docker run's container
# ID must reach stdout; everything else must go to stderr via log().

# ── Server lifecycle ──────────────────────────────────────────────────────────
start_server() {
    local name="$1" image="$2" port="$3" kv_dtype="$4"
    local block_arg="" dtype_val="auto"

    if [[ "$kv_dtype" == kvarn_* ]]; then
        block_arg="--block-size 128"
        # KVarN runs in float16 compute (README requirement). For FP8 weight models
        # this dequantizes weights to float16, which avoids the FP8 kernel path that
        # conflicts with KVarN's Triton attention backend.
        dtype_val="float16"
    fi

    log "Starting $name on :$port  [kv_cache_dtype=$kv_dtype  dtype=$dtype_val]"

    docker run -d --rm \
        --gpus all \
        --ipc=host \
        -p "$port:8080" \
        -v "$MODEL_DIR:/app/models:z" \
        -e HOST=0.0.0.0 \
        -e PORT=8080 \
        -e MODEL_NAME="$MODEL" \
        -e DTYPE="$dtype_val" \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        -e GPU_MEMORY_UTILIZATION="$GPU_MEM" \
        -e KV_CACHE_DTYPE="$kv_dtype" \
        -e EXTRA_ARGS="$SHARED_EXTRA_ARGS $block_arg" \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$image"
}

wait_healthy() {
    local port="$1" label="$2" timeout=720
    log "Waiting for $label to become healthy (up to ${timeout}s)..."
    for i in $(seq 1 $timeout); do
        if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
            log "$label is healthy"
            return 0
        fi
        sleep 1
    done
    log "Logs from failed container:"
    docker logs "$CID" 2>&1 | tail -40 || true
    fail "$label did not become healthy after ${timeout}s"
}

stop_container() {
    local cid="${1:-}"
    [[ -z "$cid" ]] && return
    docker stop "$cid" >/dev/null 2>&1 || true
}

# ── KV cache pool size from /metrics ─────────────────────────────────────────
kv_pool_info() {
    local port="$1"
    local usage blocks
    usage=$(curl -s "http://localhost:$port/metrics" 2>/dev/null \
        | grep "vllm:gpu_cache_usage_perc" | awk '{print $NF}' || echo "n/a")
    blocks=$(curl -s "http://localhost:$port/metrics" 2>/dev/null \
        | grep "vllm:num_gpu_blocks{" | awk '{print $NF}' | head -1 || echo "n/a")
    echo "usage_pct=$usage  gpu_blocks=$blocks"
}

# ── Resolve served model name (vLLM serves under local path, not HF ID) ───────
get_served_model() {
    local port="$1"
    curl -sf "http://localhost:$port/v1/models" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null \
        || echo "$MODEL"
}

# ── Throughput benchmark ───────────────────────────────────────────────────────
run_throughput() {
    local label="$1" port="$2" outfile="$3"
    log "Throughput sweep: $label"

    local served_model
    served_model=$(get_served_model "$port")
    log "  served model name: $served_model"

    {
        echo "=== $label ==="
        echo "model=$served_model  kv_dtype=$4  max_model_len=$MAX_MODEL_LEN"
        echo "timestamp=$(date -Iseconds)"
        echo ""
    } > "$outfile"

    local last_ok=0
    for np in $CONCURRENCY_LEVELS; do
        log "  concurrency=$np ..."

        # openai backend = /v1/completions (no endpoint suffix confusion).
        # --tokenizer points to host-side path because the benchmark client runs
        # on the host but the server reports a Docker-internal path as model name.
        result=$(vllm bench serve \
            --backend openai \
            --host localhost \
            --port "$port" \
            --model "$served_model" \
            --tokenizer "$MODEL_DIR/$MODEL" \
            --dataset-name random \
            --num-prompts "$np" \
            --max-concurrency "$np" \
            --output-len "$NUM_OUTPUT_TOKENS" \
            --input-len 512 \
            --ignore-eos \
            --num-warmups 1 \
            --percentile-metrics ttft,tpot,e2el \
            2>&1 || echo "FAILED")

        echo "--- concurrency=$np ---" >> "$outfile"
        echo "$result" >> "$outfile"
        echo "" >> "$outfile"

        if echo "$result" | grep -qiE "out.of.memory|cuda error|killed|oom"; then
            log "  OOM at concurrency=$np — stopping sweep"
            echo "OOM_AT_CONCURRENCY=$np" >> "$outfile"
            break
        fi
        last_ok=$np
    done
    echo "MAX_SUCCESSFUL_CONCURRENCY=$last_ok" >> "$outfile"
}

# ── Accuracy spot-check ────────────────────────────────────────────────────────
PROMPTS=(
    "What is the capital of France?"
    "What is 17 multiplied by 23?"
    "Who wrote the play Hamlet?"
    "What element has the atomic number 79?"
    "Briefly name Newton's three laws of motion."
)

run_accuracy() {
    local label="$1" port="$2" outfile="$3"
    log "Accuracy check: $label"

    local served_model
    served_model=$(get_served_model "$port")

    {
        echo "=== $label ==="
        echo "model=$served_model  kv_dtype=$4"
        echo ""
    } > "$outfile"

    for i in "${!PROMPTS[@]}"; do
        local prompt="${PROMPTS[$i]}"
        local response
        response=$(curl -sf "http://localhost:$port/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$served_model\",
                \"prompt\": \"$prompt\",
                \"max_tokens\": 200,
                \"temperature\": 0
            }" \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'].strip())" \
            2>/dev/null || echo "(request failed)")

        echo "Q$((i+1)): $prompt" >> "$outfile"
        echo "A:  $response" >> "$outfile"
        echo "" >> "$outfile"
    done
}

# ── Main ──────────────────────────────────────────────────────────────────────
ACCURACY_ONLY=0
THROUGHPUT_ONLY=0
for arg in "$@"; do
    [[ "$arg" == "--accuracy-only"   ]] && ACCURACY_ONLY=1
    [[ "$arg" == "--throughput-only" ]] && THROUGHPUT_ONLY=1
done

CID=""
cleanup() { stop_container "$CID"; }
trap cleanup EXIT

# ── Phase 1: Baseline (FP8 KV cache) ─────────────────────────────────────────
log "════════════════════════════════════════"
log " Phase 1: Baseline  [kv=$BASELINE_KV_DTYPE]"
log "════════════════════════════════════════"

CID=$(start_server "baseline" "$BASELINE_IMAGE" "$PORT_BASE" "$BASELINE_KV_DTYPE")
wait_healthy "$PORT_BASE" "baseline"

log "Baseline KV pool: $(kv_pool_info $PORT_BASE)"

[[ $ACCURACY_ONLY  -eq 0 ]] && run_throughput "baseline-fp8"  "$PORT_BASE" "$RESULTS_DIR/throughput_baseline.txt" "$BASELINE_KV_DTYPE"
[[ $THROUGHPUT_ONLY -eq 0 ]] && run_accuracy   "baseline-fp8"  "$PORT_BASE" "$RESULTS_DIR/accuracy_baseline.txt"   "$BASELINE_KV_DTYPE"

stop_container "$CID"; CID=""

# ── Phase 2: KVarN ───────────────────────────────────────────────────────────
log "════════════════════════════════════════"
log " Phase 2: KVarN  [kv=$KVARN_KV_DTYPE]"
log "════════════════════════════════════════"

CID=$(start_server "kvarn" "$KVARN_IMAGE" "$PORT_KVARN" "$KVARN_KV_DTYPE")
wait_healthy "$PORT_KVARN" "kvarn"

log "KVarN KV pool: $(kv_pool_info $PORT_KVARN)"

[[ $ACCURACY_ONLY  -eq 0 ]] && run_throughput "kvarn-k4v2"  "$PORT_KVARN" "$RESULTS_DIR/throughput_kvarn.txt" "$KVARN_KV_DTYPE"
[[ $THROUGHPUT_ONLY -eq 0 ]] && run_accuracy   "kvarn-k4v2"  "$PORT_KVARN" "$RESULTS_DIR/accuracy_kvarn.txt"   "$KVARN_KV_DTYPE"

stop_container "$CID"; CID=""

# ── Report ────────────────────────────────────────────────────────────────────
log "════════════════════════════════════════"
log " Results"
log "════════════════════════════════════════"

if [[ $ACCURACY_ONLY -eq 0 ]]; then
    echo ""
    echo "── Throughput summary ──"
    echo ""
    echo "Baseline (fp8_e4m3):"
    grep -E "output_throughput|request_throughput|MAX_SUCCESSFUL|OOM_AT" \
        "$RESULTS_DIR/throughput_baseline.txt" 2>/dev/null || echo "  (no data)"
    echo ""
    echo "KVarN (k4v2_g128):"
    grep -E "output_throughput|request_throughput|MAX_SUCCESSFUL|OOM_AT" \
        "$RESULTS_DIR/throughput_kvarn.txt" 2>/dev/null || echo "  (no data)"
fi

if [[ $THROUGHPUT_ONLY -eq 0 ]]; then
    echo ""
    echo "── Accuracy diff (baseline → KVarN) ──"
    diff "$RESULTS_DIR/accuracy_baseline.txt" "$RESULTS_DIR/accuracy_kvarn.txt" || true
fi

echo ""
log "Full results: $RESULTS_DIR"
