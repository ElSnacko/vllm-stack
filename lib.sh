#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_error()   { echo -e "${RED}Error:${NC} $1" >&2; }
log_info()    { echo -e "${GREEN}Info:${NC} $1" >&2; }
log_warning() { echo -e "${YELLOW}Warning:${NC} $1" >&2; }
log_success() { echo -e "${GREEN}Success:${NC} $1" >&2; }

format_size() {
    local bytes=$1
    if [ "$bytes" -ge 1073741824 ]; then
        awk -v b="$bytes" 'BEGIN {printf "%.1f GB", b/1024/1024/1024}'
    elif [ "$bytes" -ge 1048576 ]; then
        awk -v b="$bytes" 'BEGIN {printf "%.1f MB", b/1024/1024}'
    elif [ "$bytes" -ge 1024 ]; then
        awk -v b="$bytes" 'BEGIN {printf "%.1f KB", b/1024}'
    else
        echo "${bytes} B"
    fi
}

detect_active_version() {
    local versions_dir="${1:-$(pwd)/versions}"
    if [ -f "${versions_dir}/active" ]; then
        cat "${versions_dir}/active"
    else
        echo "unknown"
    fi
}

set_active_version() {
    local version="$1"
    local versions_dir="${2:-$(pwd)/versions}"
    mkdir -p "$versions_dir"
    echo "$version" > "${versions_dir}/active"
}

version_needs_pull() {
    local active_version="$1"
    local image="vllm/vllm-openai:${active_version}"
    if docker image inspect "$image" &>/dev/null; then
        return 1
    fi
    return 0
}

check_deps() {
    local missing=()
    command -v docker &>/dev/null || missing+=("docker")
    docker compose version &>/dev/null 2>&1 || missing+=("docker-compose (plugin)")
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        echo "Install with: sudo apt install docker.io docker-compose-plugin" >&2
        exit 1
    fi
}

check_nvidia() {
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        log_warning "NVIDIA Container Toolkit not detected — GPU passthrough may not work"
        log_warning "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
}

check_build_deps() {
    local missing=()
    command -v curl &>/dev/null || missing+=("curl")
    command -v jq &>/dev/null || missing+=("jq")
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        echo "Install with: sudo apt install ${missing[*]}" >&2
        exit 1
    fi
}

set_env_var() {
    local key="$1"
    local value="$2"
    local env_file="${3:-.env}"

    local escaped_value
    escaped_value=$(printf '%s\n' "$value" | sed 's/[&/\]/\\&/g')
    if grep -q "^${key}=" "$env_file"; then
        sed -i "s|^${key}=.*|${key}=${escaped_value}|" "$env_file"
    else
        echo "${key}=${value}" >> "$env_file"
    fi
}

set_model_in_env() {
    set_env_var "MODEL_NAME" "$1" "${2:-.env}"
}

source_env_files() {
    [ -f .env ] || { [ -f .env.example ] && cp .env.example .env; }
    [ -f gpu.env ] || { [ -f gpu.env.example ] && cp gpu.env.example gpu.env; }
    [ -f vllm.args ] || { [ -f vllm.args.example ] && cp vllm.args.example vllm.args; }

    if [ ! -f .env ]; then
        log_error "No .env file found and no .env.example to copy from"
        exit 1
    fi

    set -a
    source .env
    if [ -f gpu.env ]; then
        source gpu.env
    fi
    set +a
}

enumerate_models() {
    local model_dir="${1:-./llm_models/hf}"
    if [ ! -d "$model_dir" ]; then
        return 1
    fi

    find "$model_dir" -maxdepth 2 -mindepth 2 -type d ! -path '*/\.*' \
        | sort | while read -r d; do
        local rel="${d#${model_dir}/}"

        local size=0
        local has_config=false
        local has_safetensors=false
        local has_bin=false
        local file_count=0

        for f in "$d"/*; do
            [ -f "$f" ] || continue
            file_count=$((file_count + 1))
            local s
            s=$(stat -c%s "$f" 2>/dev/null || echo 0)
            size=$((size + s))
            case "$(basename "$f")" in
                config.json)    has_config=true ;;
                *.safetensors)  has_safetensors=true ;;
                *.bin|*.pt)     has_bin=true ;;
            esac
        done

        local model_type="unknown"
        if [ "$has_safetensors" = true ]; then
            model_type="safetensors"
        elif [ "$has_bin" = true ]; then
            model_type="pytorch"
        fi

        local is_valid="false"
        if [ "$has_config" = true ] && [ $file_count -gt 1 ]; then
            is_valid="true"
        fi

        echo "${size}|${rel}|${model_type}|${is_valid}|${file_count}"
    done
}

select_model_interactive() {
    local lines
    lines=$(enumerate_models)
    if [ -z "$lines" ]; then
        echo "No HuggingFace models found." >&2
        return 1
    fi

    local count=0
    local -a files=()

    echo "Select a model:" >&2
    echo "" >&2

    while IFS='|' read -r size rel model_type is_valid file_count; do
        count=$((count + 1))
        files+=("$rel")
        local valid_tag=""
        [ "$is_valid" = "false" ] && valid_tag=" [INCOMPLETE]"
        printf "  %2d) %-55s %5s GB  %s%s\n" "$count" "$rel" "$(awk -v s="$size" 'BEGIN {printf "%.1f", s/1024/1024/1024}')" "$model_type" "$valid_tag" >&2
    done <<< "$lines"

    echo "" >&2
    read -p "  Choice (1-${count}): " selection >&2

    if [[ $selection =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le "$count" ]; then
        echo "${files[$((selection-1))]}"
    else
        echo "Invalid selection." >&2
        return 1
    fi
}
