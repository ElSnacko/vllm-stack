#!/bin/bash
exec python3 "$(dirname "${BASH_SOURCE[0]}")/download_model.py" "$@"
