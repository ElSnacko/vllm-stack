#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

source lib.sh

MODEL_DIR="./llm_models/hf"
MODE="list"
TARGET=""

show_help() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Manage downloaded HuggingFace models.

Commands:
  list              List all models with sizes (default)
  info [PATTERN]    Show detailed info for model matching PATTERN
  delete [PATTERN]  Delete model files matching PATTERN
  help              Show this help

Examples:
  $0                                List all models
  $0 list                           List all models
  $0 info Qwen                      Show info for models matching "Qwen"
  $0 delete meta-llama/old-model    Delete specific model
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        list|ls)       MODE="list"; shift ;;
        info|i)        MODE="info"; TARGET="${2:-}"; shift 2 ;;
        delete|del|rm) MODE="delete"; TARGET="${2:-}"; shift 2 ;;
        help|--help|-h) show_help; exit 0 ;;
        *)             MODE="info"; TARGET="$1"; shift ;;
    esac
done

if [ ! -d "$MODEL_DIR" ]; then
    echo "No models directory found. Download models with: ./download_model.sh"
    exit 1
fi

LINES=$(enumerate_models "$MODEL_DIR")

if [ "$MODE" = "list" ]; then
    if [ -z "$LINES" ]; then
        echo "No models found in ${MODEL_DIR}"
        exit 0
    fi

    printf "  %-55s %10s  %-12s %s\n" "MODEL" "SIZE" "TYPE" "STATUS"
    echo "  $(printf '%0.s-' {1..90})"

    while IFS='|' read -r size rel model_type is_valid file_count; do
        status="OK"
        [ "$is_valid" = "false" ] && status="INCOMPLETE"
        printf "  %-55s %10s  %-12s %s\n" "$rel" "$(format_size "$size")" "$model_type" "$status"
    done <<< "$LINES"

    echo ""
    total=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
    echo "  Total disk usage: ${total}"
fi

if [ "$MODE" = "info" ]; then
    if [ -z "$TARGET" ]; then
        echo "Specify a model name or path pattern. Use '$0 list' to see all models."
        exit 1
    fi

    local_lines=$(echo "$LINES" | grep -iF "$TARGET" || true)
    if [ -z "$local_lines" ]; then
        echo "No models matching '${TARGET}'"
        exit 1
    fi

    while IFS='|' read -r size rel model_type is_valid file_count; do
        echo "  Name:    ${rel}"
        echo "  Path:    ${MODEL_DIR}/${rel}"
        echo "  Size:    $(format_size "$size")"
        echo "  Type:    ${model_type}"
        echo "  Files:   ${file_count}"
        echo "  Status:  $([ "$is_valid" = "true" ] && echo "complete" || echo "incomplete")"

        model_path="${MODEL_DIR}/${rel}"
        if [ -f "${model_path}/config.json" ]; then
            arch=$(python3 -c '
import json, sys
try:
    c = json.load(open(sys.argv[1]))
    print(c.get("model_type", c.get("architectures", ["unknown"])[0] if "architectures" in c else "unknown"))
except: print("unknown")
' "${model_path}/config.json" 2>/dev/null || echo "unknown")
            echo "  Arch:    ${arch}"
        fi
        echo ""
    done <<< "$local_lines"
fi

if [ "$MODE" = "delete" ]; then
    if [ -z "$TARGET" ]; then
        echo "Specify a model to delete. Use '$0 list' to see all models."
        exit 1
    fi

    local_lines=$(echo "$LINES" | grep -iF "$TARGET" || true)
    if [ -z "$local_lines" ]; then
        echo "No models matching '${TARGET}'"
        exit 1
    fi

    match_count=$(echo "$local_lines" | wc -l)
    echo "Matching models (${match_count}):"
    echo ""
    while IFS='|' read -r size rel model_type is_valid file_count; do
        printf "  %-55s %10s\n" "$rel" "$(format_size "$size")"
    done <<< "$local_lines"
    echo ""

    read -p "Delete all ${match_count} matching model(s)? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi

    while IFS='|' read -r size rel model_type is_valid file_count; do
        [ -z "$rel" ] && continue
        [ -d "${MODEL_DIR}/${rel}" ] || continue
        echo "Removing: ${rel}"
        rm -rf "${MODEL_DIR}/${rel}"
    done <<< "$local_lines"

    rmdir "$MODEL_DIR" 2>/dev/null || true
    echo "Done."
fi
