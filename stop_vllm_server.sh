#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

source lib.sh

STOP_ALL=false
STOP_PORT=""

show_help() {
    echo "Usage: $0 [PORT] [--all] [--help]"
    echo ""
    echo "Stop a vllm-server instance."
    echo ""
    echo "Arguments:"
    echo "  PORT          Stop instance on specific port (default: base port from .env)"
    echo "  --all         Stop all running instances"
    echo "  --help, -h    Show this help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --all|-a)  STOP_ALL=true; shift ;;
        --help|-h) show_help; exit 0 ;;
        -*)        echo "Unknown option: $1. Use --help for usage."; exit 1 ;;
        *)         STOP_PORT="$1"; shift ;;
    esac
done

stop_instance() {
    local port="$1"
    local env_file=".env"
    if [ "$port" != "$BASE_PORT" ]; then
        env_file=".env.${port}"
    fi
    local project="vllm-${port}"

    if ! docker compose -p "$project" ps -q vllm-server 2>/dev/null | grep -q .; then
        echo "vllm-server :${port} is not running."
        return 1
    fi

    docker compose -p "$project" down 2>/dev/null
    echo "vllm-server :${port} stopped."
    return 0
}

BASE_PORT="8080"
if [ -f .env ]; then
    BP=$(grep -E "^HOST_PORT=" .env 2>/dev/null | tail -1 | cut -d= -f2-)
    [ -n "$BP" ] && BASE_PORT="$BP"
fi

if [ "$STOP_ALL" = true ]; then
    FOUND=0
    stop_instance "$BASE_PORT" && FOUND=1 || true
    for env_file in .env.*; do
        [ -f "$env_file" ] || continue
        port="${env_file#.env.}"
        [[ "$port" =~ ^[0-9]+$ ]] || continue
        stop_instance "$port" && FOUND=1 || true
    done
    if [ "$FOUND" -eq 0 ]; then
        echo "No running instances found."
    fi
else
    PORT="${STOP_PORT:-$BASE_PORT}"
    stop_instance "$PORT"
fi
