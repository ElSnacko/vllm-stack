# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A local vLLM server management system for running high-performance LLM inference with NVIDIA CUDA on an RTX 5090 (32 GB VRAM, Blackwell architecture). Provides a Dockerized environment with multi-instance support, model management via HuggingFace, and version management for vLLM releases.

**Target hardware: NVIDIA RTX 5090 workstation (not the current AMD machine).**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Management Scripts                        │
│  run_vllm_server.sh, select_model.sh, stop_vllm_server.sh  │
│  switch_version.sh, update_version.sh, manage_models.sh    │
│  download_model.sh, bench_vllm.sh                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 Docker Containers (multi-instance)           │
│  vllm-<port>-vllm-server-1 (from vllm/vllm-openai image)   │
│  - NVIDIA CUDA GPU via nvidia-container-toolkit             │
│  - Models mounted from llm_models/hf/                       │
│  - Each port gets its own config (.env.<port>) and container│
└─────────────────────────────────────────────────────────────┘
```

### Key Components

**Shared Library** (`lib.sh`):
- Color/logging: `log_error()`, `log_info()`, `log_warning()`, `log_success()` (all write to stderr)
- `format_size()`, `detect_active_version()`, `set_active_version()`
- `version_needs_pull()` — checks if Docker image exists locally
- `check_deps()`, `check_nvidia()`, `check_build_deps()` — verify required tools
- `set_env_var()`, `set_model_in_env()` — edit .env files
- `enumerate_models()` — scan `llm_models/hf/` for downloaded HuggingFace models
- `select_model_interactive()` — interactive model picker
- `source_env_files()` — loads .env, gpu.env, vllm.args

**Management Scripts** (in project root):
- `run_vllm_server.sh` - Starts Docker container with vLLM OpenAI-compatible server; supports multi-instance via `--port`
  - `--port, -p PORT` — server port (enables multi-instance; each port gets `.env.<port>`)
  - `--context, -c SIZE` — set max model length (token count)
  - `--gpu-util FLOAT` — GPU memory utilization (0.0-1.0, default 0.95)
  - `--dtype DTYPE` — data type: auto, float16, bfloat16, float8
  - `--tp N` — tensor parallel size
  - `--rebuild` — force Docker image rebuild
  - `--status` — show all instances (or specific with `--port`)
  - `--wait SECONDS` — max time to wait for health
- `select_model.sh` - Interactive/non-interactive model selection (supports --list, --select N)
- `stop_vllm_server.sh` - Stop instances: `stop_vllm_server.sh [PORT]` or `--all`
- `switch_version.sh` - Switch between vLLM Docker image versions (reads/writes `versions/active`)
- `update_version.sh` - Pull new vLLM Docker images from Docker Hub
- `manage_models.sh` - List, inspect, and delete downloaded models
- `download_model.sh` / `download_model.py` - Download models from HuggingFace (safetensors/pytorch format)
- `bench_vllm.sh` - Benchmark running vLLM server via OpenAI API

**Version Storage** (`versions/`):
- `versions/active` - Plain text file containing the active Docker image tag (e.g., `v0.8.5`, `latest`)

**Model Storage** (`llm_models/hf/`):
- Organized as `org/model-name/` (HuggingFace format, not GGUF)
- Contains safetensors/bin weight files, config.json, tokenizer files

**Configuration**:
- `.env` - Server config for default instance (model, max-model-len, gpu-util, port)
- `.env.<port>` - Per-instance config for non-default ports (copied from `.env` on first run)
- `gpu.env` - GPU driver settings (CUDA_VISIBLE_DEVICES, VLLM_USE_V1, worker settings)
- `vllm.args` - Extra vLLM arguments (one per line, # for comments; shared across instances)

## Docker Configuration

The Dockerfile:
- Based on `vllm/vllm-openai:${VLLM_VERSION}` (official vLLM image)
- Adds curl (for health checks)
- Stamps version as `vllm.version` label
- Exposes port 8080
- Entrypoint: `entrypoint.sh` (assembles args, launches `vllm.entrypoints.openai.api_server`)

Docker run configuration (`docker-compose.yml`):
- GPU access via `deploy.resources.reservations.devices` (NVIDIA Container Toolkit)
- Volume mount: `./llm_models/hf:/app/models`
- HuggingFace cache: `~/.cache/huggingface:/root/.cache/huggingface`
- Health check on `/health` endpoint
- `shm_size: 4g` (vLLM requires large shared memory)

## Multi-Instance Architecture

Each port runs an isolated Docker Compose project (`vllm-<port>`):

| Component | Default instance | Additional instance |
|---|---|---|
| Port | from `.env` (default 8080) | via `--port` CLI flag |
| Config file | `.env` | `.env.<port>` (copied from `.env` on first run) |
| Docker project | `vllm-8080` | `vllm-8081` |
| Container name | `vllm-8080-vllm-server-1` | `vllm-8081-vllm-server-1` |

All instances share:
- The same Docker image (`vllm-server:latest`)
- `gpu.env` and `vllm.args` (mounted read-only into all containers)
- The `llm_models/hf/` model directory

## Version State Tracking

Single source of truth: `versions/active` file (plain text with the Docker tag).
- `switch_version.sh` updates the file and optionally rebuilds
- `run_vllm_server.sh` reads it for Docker build arg

## RTX 5090 Notes

- **32 GB VRAM** — can fit most 30B-parameter models at BF16, or 70B at FP8/AWQ
- **Blackwell SM 12.0** — supports FP8 natively; use `--dtype float8` or `--kv-cache-dtype fp8` for max throughput
- **VLLM_USE_V1=1** — enable the V1 engine for better scheduling
- **GPU_MEMORY_UTILIZATION=0.95** — leave 5% headroom for CUDA overhead; adjust down if OOMs
- **Tensor parallelism** — single GPU, so `TENSOR_PARALLEL_SIZE=1`; not applicable unless multi-GPU

## Development Commands

```bash
./run_vllm_server.sh              # Start server (waits for health)
./run_vllm_server.sh --status     # Show all instances (version/model/port/health)
./run_vllm_server.sh --rebuild    # Force image rebuild
./run_vllm_server.sh -p 8081 -m   # Start second instance on port 8081
./run_vllm_server.sh -p 8081 --context 4096  # Override context for instance
./run_vllm_server.sh --status -p 8081        # Status for one instance
./stop_vllm_server.sh             # Stop default instance
./stop_vllm_server.sh 8081        # Stop instance on port 8081
./stop_vllm_server.sh --all       # Stop all instances
./switch_version.sh --list        # List locally pulled versions
./switch_version.sh v0.8.5        # Switch to specific version
./update_version.sh --list        # List available Docker Hub tags
./update_version.sh --latest      # Pull latest
./update_version.sh --notes v0.8.5  # Show release info
./manage_models.sh list           # List downloaded models
./download_model.sh Qwen/Qwen3-30B-A3B --all   # Download model
./download_model.sh --login                        # Save HF token
./bench_vllm.sh                   # Run benchmarks
```

## Model Path Resolution

1. `download_model.py` downloads from HuggingFace to `llm_models/hf/org/model/`
2. `select_model.sh` scans `llm_models/hf/` for directories with `config.json`
3. User selects interactively, or non-interactively with `--select N`
4. Script outputs `org/model` path
5. `run_vllm_server.sh` captures this, saves to `.env` as `MODEL_NAME`
6. Container sees model at `/app/models/org/model/`

## Differences from the llama.cpp Stack

| Aspect | llama.cpp stack | vLLM stack |
|---|---|---|
| Model format | GGUF | HuggingFace (safetensors/bin) |
| GPU backend | Vulkan / ROCm | NVIDIA CUDA |
| Build management | Pre-built binaries in `builds/` | Docker Hub image tags in `versions/` |
| Server binary | `llama-server` | `vllm.entrypoints.openai.api_server` |
| GPU access | `/dev/dri`, `--group-add video` | `deploy.resources.reservations.devices` (nvidia) |
| GPU config | `RADV_PERFTEST`, `VK_ICD_FILENAMES` | `CUDA_VISIBLE_DEVICES`, `VLLM_USE_V1` |
| shm_size | 1g | 4g (vLLM requirement) |
| Health start | 120s | 300s (model loading is slower) |
