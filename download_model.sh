#!/bin/bash
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is required but not installed." >&2
    echo "Install with: sudo apt install python3" >&2
    exit 1
fi
exec python3 "$(dirname "${BASH_SOURCE[0]}")/download_model.py" "$@"
