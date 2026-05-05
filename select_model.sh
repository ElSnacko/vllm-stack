#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

source lib.sh

MODEL_DIR="./llm_models/hf"
MODE="interactive"
SELECT_NUM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --list|-l)    MODE="list"; shift ;;
        --select|-s)  MODE="select"; SELECT_NUM="${2:-}"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --list, -l       List models (non-interactive)"
            echo "  --select, -s N   Select Nth model (non-interactive)"
            echo "  --help, -h       Show this help"
            echo ""
            echo "Without options: interactive model selection"
            exit 0
            ;;
        *) echo "Unknown option: $1. Use --help for usage." >&2; exit 1 ;;
    esac
done

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory $MODEL_DIR does not exist." >&2
    echo "Download models first with: ./download_model.sh" >&2
    exit 1
fi

LINES=$(enumerate_models "$MODEL_DIR")
if [ -z "$LINES" ]; then
    echo "No HuggingFace models found in $MODEL_DIR" >&2
    exit 1
fi

COUNT=0
declare -a MODEL_RELS=()

while IFS='|' read -r size rel model_type is_valid file_count; do
    COUNT=$((COUNT + 1))
    MODEL_RELS+=("$rel")
done <<< "$LINES"

print_model_list() {
    local i=0
    while IFS='|' read -r size rel model_type is_valid file_count; do
        i=$((i + 1))
        local valid_tag=""
        [ "$is_valid" = "false" ] && valid_tag=" [INCOMPLETE]"
        printf "  %2d) %-55s %5s GB  %s%s\n" "$i" "$rel" "$(awk -v s="$size" 'BEGIN {printf "%.1f", s/1024/1024/1024}')" "$model_type" "$valid_tag"
    done <<< "$LINES"
}

if [ "$MODE" = "list" ]; then
    print_model_list
    exit 0
fi

if [ "$MODE" = "select" ]; then
    if [ -z "$SELECT_NUM" ]; then
        echo "Error: --select requires a number." >&2
        exit 1
    fi
    if [[ "$SELECT_NUM" =~ ^[0-9]+$ ]] && [ "$SELECT_NUM" -ge 1 ] && [ "$SELECT_NUM" -le "$COUNT" ]; then
        echo "${MODEL_RELS[$((SELECT_NUM-1))]}"
    else
        echo "Invalid selection: $SELECT_NUM (must be 1-${COUNT})" >&2
        exit 1
    fi
    exit 0
fi

echo "Select a model:" >&2
echo "" >&2
print_model_list >&2

echo "" >&2
read -p "  Choice (1-${COUNT}): " selection >&2

if [[ $selection =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le "$COUNT" ]; then
    echo "${MODEL_RELS[$((selection-1))]}"
else
    echo "Invalid selection." >&2
    exit 1
fi
