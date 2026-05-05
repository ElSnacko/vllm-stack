#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

source lib.sh

error()   { log_error "$1"; exit 1; }
info()    { log_info "$1"; }
warning() { log_warning "$1"; }
success() { log_success "$1"; }

VLLM_DOCKER_REPO="vllm/vllm-openai"
DOCKER_HUB_API="https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags"

show_help() {
    cat << EOF
Pull vLLM Docker images from Docker Hub

Usage:
    $0                                    Interactive selection
    $0 --latest                           Pull latest image
    $0 --version <tag>                    Pull specific version (e.g. v0.8.5)
    $0 --list                             List available tags from Docker Hub
    $0 --list-local                       List locally pulled images
    $0 --cleanup                          Remove old local images
    $0 --notes <version>                  Show release info from GitHub
    $0 --help                             Show this help

Examples:
    $0 --latest                           Pull latest vLLM
    $0 --version v0.8.5                   Pull specific version
    $0 --version latest                   Pull latest tag explicitly
    $0 --list                             Browse available versions

Note: After pulling, switch to the new version:
    ./switch_version.sh <tag>
    ./run_vllm_server.sh --rebuild
EOF
}

fetch_tags() {
    local page_size="${1:-100}"
    local url="${DOCKER_HUB_API}?page_size=${page_size}&ordering=last_updated"

    local response
    response=$(curl -s "$url" 2>/dev/null || echo "")
    if [ -z "$response" ]; then
        error "Failed to fetch tags from Docker Hub. Check network connection."
        return 1
    fi

    echo "$response" | jq -r '.results[] | [.name, .full_size, .last_updated] | @json' 2>/dev/null \
        | sort -t'"' -k2 | {
        local ordered=""
        local rest=""
        while IFS= read -r line; do
            local tag
            tag=$(echo "$line" | jq -r '.[0]')
            if [[ "$tag" =~ ^v[0-9] ]]; then
                ordered+="${line}"$'\n'
            else
                rest+="${line}"$'\n'
            fi
        done
        printf '%s%s' "$ordered" "$rest"
    }
}

list_tags() {
    local tags
    tags=$(fetch_tags)

    if [ -z "$tags" ]; then
        error "No tags found"
        return 1
    fi

    local current_version
    current_version=$(detect_active_version)

    echo ""
    echo "Available vLLM Docker image tags:"
    echo ""

    local count=1
    while IFS= read -r line; do
        [ -z "$line" ] && continue

        local tag size updated
        tag=$(echo "$line" | jq -r '.[0]')
        size=$(echo "$line" | jq -r '.[1]')
        updated=$(echo "$line" | jq -r '.[2]')

        local formatted_date
        formatted_date=$(date -d "$updated" +%Y-%m-%d 2>/dev/null || echo "$updated" | cut -dT -f1)

        local marker=""
        if [ "$tag" = "$current_version" ]; then
            marker=" ${GREEN}[ACTIVE]${NC}"
        fi

        local size_human
        size_human=$(format_size "${size:-0}")

        local local_marker=""
        if docker image inspect "${VLLM_DOCKER_REPO}:${tag}" &>/dev/null; then
            local_marker=" ${CYAN}[PULLED]${NC}"
        fi

        printf "  ${GREEN}%2d${NC}. %-20s %10s  %s%b%b\n" \
            "$count" "$tag" "$size_human" "$formatted_date" "$marker" "$local_marker"

        ((count++))
    done <<< "$tags"

    echo ""

    local version_tags=()
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        local tag
        tag=$(echo "$line" | jq -r '.[0]')
        [[ "$tag" =~ ^v[0-9] ]] && [[ "$tag" != *-* ]] && version_tags+=("$tag")
    done <<< "$tags"

    if [ ${#version_tags[@]} -eq 0 ]; then
        return 0
    fi

    echo "--- Release summaries ---"
    echo ""

    local release_json
    release_json=$(curl -s "https://api.github.com/repos/vllm-project/vllm/releases?per_page=${#version_tags[@]}" 2>/dev/null || echo "")

    for vtag in "${version_tags[@]}"; do
        local date body summary
        date=$(echo "$release_json" | jq -r --arg t "$vtag" '.[] | select(.tag_name == $t) | .published_at // "unknown"' 2>/dev/null)
        body=$(echo "$release_json" | jq -r --arg t "$vtag" '.[] | select(.tag_name == $t) | .body // ""' 2>/dev/null)

        if [ -n "$body" ]; then
            summary=$(echo "$body" | tr -d '\r' | sed '/^#/d; /^$/d; /^```/d; /^---/d' | head -1 | cut -c1-120) || true
        fi

        formatted_date=""
        if [ "$date" != "unknown" ] && [ -n "$date" ]; then
            formatted_date="($(date -d "$date" +%Y-%m-%d 2>/dev/null))"
        fi

        if [ -n "$summary" ]; then
            printf "  ${GREEN}%-15s${NC}  %-14s %s\n" "$vtag" "$formatted_date" "$summary"
        elif [ -n "$date" ] && [ "$date" != "unknown" ]; then
            printf "  ${GREEN}%-15s${NC}  %-14s\n" "$vtag" "$formatted_date"
        fi
    done

    echo ""
    echo "  Full changelog: ./update_version.sh --notes <version>"
    echo ""
}

list_local() {
    echo ""
    echo "Locally pulled vLLM images:"
    echo ""

    local images
    images=$(docker images --format "{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" \
        | grep "vllm/vllm-openai:" \
        | sort -t: -k2 -r || true)

    if [ -z "$images" ]; then
        echo "  No vLLM images pulled yet."
        echo "  Pull one with: $0 --latest"
        return
    fi

    while IFS=$'\t' read -r tag size created; do
        printf "  %-35s %10s  %s\n" "$tag" "$size" "$created"
    done <<< "$images"

    echo ""
}

pull_image() {
    local tag="$1"
    local image="${VLLM_DOCKER_REPO}:${tag}"

    info "Pulling ${image}..."
    echo ""

    if docker pull "$image"; then
        success "Pulled: ${image}"
    else
        error "Pull failed for ${image}"
    fi
}

cleanup_old_images() {
    local current_version
    current_version=$(detect_active_version)

    info "Cleaning up old vLLM images (keeping ${current_version})..."

    local images
    images=$(docker images --format "{{.Repository}}:{{.Tag}}" \
        | grep "vllm/vllm-openai:" || true)

    if [ -z "$images" ]; then
        info "No vLLM images to clean up"
        return
    fi

    local count=0
    while IFS= read -r tag; do
        local version="${tag#vllm/vllm-openai:}"
        if [ "$version" = "$current_version" ]; then
            continue
        fi
        info "Removing: $tag"
        docker rmi "$tag" 2>/dev/null || true
        ((count++))
    done <<< "$images"

    if [ "$count" -eq 0 ]; then
        info "No old images to remove"
    else
        success "Removed $count old image(s)"
    fi
}

show_release_info() {
    local version="$1"
    local tag_name="${version}"

    info "Fetching release info for ${version}..."

    local release_json
    release_json=$(curl -s "https://api.github.com/repos/vllm-project/vllm/releases/tags/${tag_name}" 2>/dev/null || echo "")

    if [ -z "$release_json" ] || echo "$release_json" | jq -e '.message' 2>/dev/null | grep -q "Not Found"; then
        release_json=$(curl -s "https://api.github.com/repos/vllm-project/vllm/releases" 2>/dev/null | \
            jq --arg t "$tag_name" '.[] | select(.tag_name == $t)' 2>/dev/null || echo "")
    fi

    if [ -z "$release_json" ]; then
        warning "Release info not found for $version"
        echo "  Check: https://github.com/vllm-project/vllm/releases"
        return
    fi

    local date body
    date=$(echo "$release_json" | jq -r '.published_at // "unknown"')
    body=$(echo "$release_json" | jq -r '.body // "No release notes"')

    echo ""
    echo "=== vLLM ${version} ($(date -d "$date" +%Y-%m-%d 2>/dev/null || echo "$date")) ==="
    echo ""
    echo "$body" | head -50 | sed 's/^/  /'
    echo ""
    echo "  Full notes: https://github.com/vllm-project/vllm/releases/tag/${version}"
}

main() {
    check_build_deps

    local mode="interactive"
    local target_tag=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)    show_help; exit 0 ;;
            --latest)     mode="latest"; shift ;;
            --version)    mode="version"; target_tag="$2"; shift 2 ;;
            --list)       mode="list"; shift ;;
            --list-local) mode="list-local"; shift ;;
            --cleanup)    mode="cleanup"; shift ;;
            --notes)      mode="notes"; target_tag="${2:-}"; shift; [ -n "$target_tag" ] && shift ;;
            *)            error "Unknown option: $1. Use --help for usage." ;;
        esac
    done

    case "$mode" in
        list-local)
            list_local
            ;;
        cleanup)
            cleanup_old_images
            ;;
        notes)
            if [ -z "$target_tag" ]; then
                error "Specify a version: $0 --notes v0.8.5"
            fi
            show_release_info "$target_tag"
            ;;
        list)
            list_tags
            ;;
        latest)
            pull_image "latest"
            echo ""
            echo "Switch to it:"
            echo "  ./switch_version.sh latest"
            ;;
        version)
            pull_image "$target_tag"
            echo ""
            read -p "Switch to this version now? (y/N): " switch
            if [[ "$switch" =~ ^[Yy]$ ]]; then
                set_active_version "$target_tag"
                success "Active version set to: $target_tag"
                echo ""
                echo "Rebuild and start:"
                echo "  ./run_vllm_server.sh --rebuild"
            else
                echo "Switch manually with:"
                echo "  ./switch_version.sh $target_tag"
            fi
            ;;
        interactive)
            list_tags

            read -p "Select tag to pull (or 'q' to quit): " selection
            if [[ "$selection" =~ ^[Qq]$ ]] || [ -z "$selection" ]; then
                exit 0
            fi

            if [[ "$selection" =~ ^[0-9]+$ ]]; then
                local tags
                tags=$(fetch_tags)
                local tag
                tag=$(echo "$tags" | sed -n "${selection}p" | jq -r '.[0]')
                if [ -z "$tag" ]; then
                    error "Invalid selection"
                fi
                target_tag="$tag"
            else
                target_tag="$selection"
            fi

            pull_image "$target_tag"
            echo ""
            read -p "Switch to this version? (y/N): " switch
            if [[ "$switch" =~ ^[Yy]$ ]]; then
                set_active_version "$target_tag"
                success "Active version set to: $target_tag"
                echo ""
                echo "Rebuild and start:"
                echo "  ./run_vllm_server.sh --rebuild"
            else
                echo "Switch manually with:"
                echo "  ./switch_version.sh $target_tag"
            fi
            ;;
    esac
}

main "$@"
