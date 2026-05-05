#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

source lib.sh

VERSIONS_DIR="$(pwd)/versions"

error()   { log_error "$1"; exit 1; }
info()    { log_info "$1"; }
warning() { log_warning "$1"; }
success() { log_success "$1"; }

show_help() {
    cat << EOF
Switch between vLLM Docker image versions

Usage:
    $0                       Interactive selection
    $0 --list                List available versions (local images)
    $0 --current             Show current active version
    $0 <version>             Set specific version (e.g. v0.8.5, latest)
    $0 --help                Show this help

The version is used as the Docker image tag: vllm/vllm-openai:<version>

Examples:
    $0 v0.8.5
    $0 latest
    $0 --list

Note: You must rebuild after switching: ./run_vllm_server.sh --rebuild
EOF
}

get_current_version() {
    detect_active_version
}

list_versions() {
    local current_version
    current_version=$(get_current_version)

    echo ""
    echo "Local vLLM Docker images:"
    echo ""

    local images
    images=$(docker images --format "{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" \
        | grep "vllm/vllm-openai:" \
        | sort -t: -k2 -r || true)

    if [ -z "$images" ]; then
        echo "  No vLLM images found. Pull one with: ./update_version.sh"
        return
    fi

    local index=1
    while IFS=$'\t' read -r tag size created; do
        local version="${tag#vllm/vllm-openai:}"
        local marker=""
        if [ "$version" = "$current_version" ]; then
            marker=" ${GREEN}[ACTIVE]${NC}"
        fi
        printf "  ${GREEN}%2d${NC}. %-20s %10s  %s%b\n" "$index" "$version" "$size" "$created" "$marker"
        ((index++))
    done <<< "$images"

    echo ""
}

set_version() {
    local version="$1"

    info "Setting active vLLM version to: $version"

    local tag="vllm/vllm-openai:${version}"
    if ! docker image inspect "$tag" &>/dev/null; then
        warning "Image $tag not found locally"
        read -p "Pull it now? (y/N): " pull
        if [[ "$pull" =~ ^[Yy]$ ]]; then
            info "Pulling $tag..."
            docker pull "$tag" || error "Pull failed"
        else
            warning "Image not available locally — build will fail until pulled"
        fi
    fi

    set_active_version "$version"
    success "Active version set to: $version"
    echo ""
    echo "Rebuild and start:"
    echo "  ./run_vllm_server.sh --rebuild"
}

main() {
    case $# in
        0) ;;
        1)
            if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
                show_help; exit 0
            elif [ "$1" = "--list" ]; then
                list_versions; exit 0
            elif [ "$1" = "--current" ]; then
                local current
                current=$(get_current_version)
                if [ "$current" != "unknown" ]; then
                    echo "Current active version: $current"
                    local tag="vllm/vllm-openai:${current}"
                    if docker image inspect "$tag" &>/dev/null; then
                        local size
                        size=$(docker image inspect "$tag" --format '{{.Size}}' 2>/dev/null || echo "?")
                        echo "  Image:  $tag"
                        echo "  Size:   $(format_size "$size")"
                    fi
                else
                    echo "No active version set"
                    echo "Set one with: $0 <version>"
                fi
                exit 0
            else
                set_version "$1"
                exit 0
            fi
            ;;
        *)
            error "Too many arguments. Use --help for usage."
            ;;
    esac

    list_versions

    local current
    current=$(get_current_version)
    read -p "Select version tag (or type a new tag): " selection

    if [ -z "$selection" ]; then
        echo "Cancelled."
        exit 0
    fi

    set_version "$selection"
}

main "$@"
