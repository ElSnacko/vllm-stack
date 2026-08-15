# Environment, Replication & Performance

Current production setup for this box: vLLM 0.27.0 serving `unsloth/Qwen3.8-27B-NVFP4` on a single RTX 5090.

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5090, 32 GB VRAM, Blackwell (SM 12.0, native FP4 tensor cores) |
| CPU | AMD Ryzen 7 9800X3D, 8C/8T |
| RAM | 91 GiB |
| Model storage | NVMe, ext4 (LUKS-encrypted on this box, not required) |

## Software

| Component | Version |
|---|---|
| OS | Ubuntu 24.04.4 LTS, kernel 7.0.0-28-generic |
| NVIDIA driver | 595.84 (CUDA 13.2 runtime) |
| nvidia-container-toolkit | 1.19.1-1 |
| Docker | 29.1.3 |
| Docker Compose | v2.39.4 |
| vLLM image | `vllm/vllm-openai:v0.27.0-cu129-ubuntu2404` → vLLM 0.27.0, torch 2.13.0+cu129, CUDA 12.9 |

## Model

`unsloth/Qwen3.8-27B-NVFP4` — mixed-precision `compressed-tensors` checkpoint:
- NVFP4 (4-bit) for most MLP gate/up/down projections
- FP8 dynamic-per-token for attention q/k/v/o, `lm_head`, and the last 8 MLP blocks
- MTP draft head included (`model_mtp.safetensors`) — enables speculative decoding
- Has a real vision tower (333 `model.visual.*` tensors), but vLLM 0.27.0 has no registered multimodal processor for this architecture yet, so it auto-falls-back to text-only serving. No flags needed for this — it's automatic.
- **Does not ship `chat_template.jinja` in this local copy** — fetch it separately before serving (step 4 below), or every request 400s with `"default chat template is no longer allowed"`.

## Replicating this setup

1. **Prerequisites**: NVIDIA GPU (Blackwell/sm_120 for the FP4 fast path — anything else still works but loses the speedup), driver with CUDA ≥12.9 support, `nvidia-container-toolkit` installed and configured (`nvidia-ctk runtime configure --runtime=docker`), Docker + Compose plugin.

2. **Clone this repo**, then:
   ```
   cp .env.example .env
   cp gpu.env.example gpu.env
   cp vllm.args.example vllm.args
   ```

3. **Get the model.** Download `unsloth/Qwen3.8-27B-NVFP4` (or via `./download_model.sh unsloth/Qwen3.8-27B-NVFP4 --all`) into `llm_models/hf/unsloth/Qwen3.8-27B-NVFP4/`.

4. **Fetch the chat template** (this copy's download source skipped it):
   ```
   curl -sf "https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/raw/main/chat_template.jinja" \
     -o llm_models/hf/unsloth/Qwen3.8-27B-NVFP4/chat_template.jinja
   ```

5. **Set `.env`**:
   ```
   MODEL_NAME=Qwen3.8-27B-NVFP4
   MODEL_DIR=./llm_models/hf/unsloth
   MAX_MODEL_LEN=150000
   GPU_MEMORY_UTILIZATION=0.98
   KV_CACHE_DTYPE=fp8_e4m3
   TRUST_CHAT_TEMPLATE=1
   ENFORCE_EAGER=
   ```

6. **Set `vllm.args`**:
   ```
   --reasoning-parser qwen3
   --enable-auto-tool-choice
   --tool-call-parser qwen3_xml
   --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
   --enable-prefix-caching
   --max-cudagraph-capture-size 32
   --max-num-batched-tokens 4096
   --no-enable-log-requests
   ```
   (No `--quantization` flag — vLLM auto-detects `compressed-tensors` from the checkpoint's `config.json`.)

7. **Start it**: `./run_vllm_server.sh --wait 240`

8. **Verify**: `curl localhost:8080/v1/models`, then a real chat completion (not just `/health` — that only proves the process is up, not that the template/quant kernels/parsers actually work):
   ```
   curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"<served id>","messages":[{"role":"user","content":"Say OK."}],"max_tokens":20}'
   ```

## Measured performance (this hardware, this config)

Benchmarked with `./bench_vllm.sh` against the live server, 2026-08-14.

| Metric | Result |
|---|---|
| Model weight load | 21.97 GiB VRAM |
| Total VRAM in use (idle, server up) | 31.7 / 32.6 GiB (~97%) |
| KV cache pool | 152,752 tokens |
| Max concurrency at 150,000-token requests | 1.02x (effectively single-request headroom) |
| Prompt processing (prefill), 512 tok | 5,953 tok/s |
| Prompt processing (prefill), 1024 tok | 13,128 tok/s |
| Prompt processing (prefill), 2048 tok | 26,256 tok/s |
| Token generation (decode), 128 tok | 96.8 tok/s |
| Token generation (decode), 256 tok | 106.7 tok/s |
| MTP speculative acceptance rate | ~42–72% (mean accepted length 2.3–3.2 of 3 draft tokens) |

Decode throughput (~100 tok/s) already includes the MTP speedup — the draft head proposes 3 tokens per step and roughly 2.3–3.2 are accepted on average, so real per-step cost is amortized over multiple output tokens.

## Why the parameters are set this way

**`GPU_MEMORY_UTILIZATION=0.98`** — this is a single-user box with nothing else competing for VRAM, so hand vLLM nearly the whole 32 GB. On a shared or multi-process box you'd want more headroom.

**`MAX_MODEL_LEN=150000`** (not higher) — hard VRAM ceiling. At `160000` the engine failed to start: KV cache needed 5.97 GiB but only 5.81 GiB was available after loading the 21.97 GiB weights at 0.98 utilization. vLLM itself estimated ~153,600 as the max; 150,000 leaves a small safety margin. This checkpoint has ~2 GiB less KV-cache headroom than a pure-modelopt-format NVFP4 checkpoint of similar weight size, because the FP8 attention path costs more than a uniform-NVFP4 one.

**`KV_CACHE_DTYPE=fp8_e4m3`** — FP8 KV cache roughly doubles effective cache capacity vs BF16 for the same VRAM budget, which is what makes the 150K context possible at all on 32 GB. `e4m3` (4 exponent bits, 3 mantissa) is the format Blackwell's hardware FP8 path expects.

**`ENFORCE_EAGER=` (blank, i.e. CUDA graphs ON)** — CUDA graphs precompile the forward pass and remove Python dispatch overhead between decode steps, which is where most of the ~100 tok/s decode throughput above comes from. The tradeoff is VRAM: graph capture reserves memory that would otherwise go to KV cache. It's left on here because the model fits comfortably at 150K context with graphs enabled; if you push context higher and hit an OOM specifically during graph capture (not weight loading), this is the first thing to disable.

**`--max-cudagraph-capture-size 32`** — a single-user decode batch is tiny (1 real token + 3 MTP draft tokens per step), so capturing CUDA graphs for large batch sizes just wastes VRAM reserving space for batch shapes that will never occur. Capped down from vLLM's default to reclaim that memory for KV cache.

**`--max-num-batched-tokens 4096`** — caps prefill chunk size. Smaller chunks mean lower peak activation memory during long-prompt prefill (frees more room for KV cache) at the cost of somewhat slower prefill on very long prompts. Also caps a linear-attention kernel's transient buffer size that could otherwise spike and OOM on long prompts.

**No `--quantization` flag** — letting vLLM auto-detect from `config.json`'s `quantization_config.quant_method` (`compressed-tensors` for this checkpoint) rather than forcing a value. This specific checkpoint mixes NVFP4 and FP8 per-layer; forcing a single quantization backend (e.g. `modelopt`, which is correct for other checkpoints on this box) would break weight loading entirely.

**`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`** — this checkpoint ships an MTP (Multi-Token Prediction) draft head baked into the weights. It proposes 3 draft tokens per decode step which the main model verifies in parallel; measured acceptance rate here is ~42–72%, giving a real throughput multiplier over naive one-token-at-a-time decoding without spending extra full forward passes for verification.

**`--enable-prefix-caching`** — reuses computed KV blocks across requests that share a common prefix (e.g. repeated system prompts). Essentially free for any workload with repeated context, no downside for this single-user setup.

**`--reasoning-parser qwen3` / `--tool-call-parser qwen3_xml`** — match this model family's actual output format: `<think>`-style reasoning blocks and XML-style tool calls (`<function=...><parameter=...>`), confirmed by inspecting the model's own `chat_template.jinja` rather than assumed from the model name.

**Why vLLM + NVFP4 over llama.cpp/GGUF for this hardware** — the RTX 5090 has dedicated Blackwell FP4 tensor cores. vLLM's NVFP4 kernels use them natively; llama.cpp runs standard CUDA kernels and can't yet. NVFP4's calibrated per-weight rounding (used by NVIDIA's modelopt/compressed-tensors quantizers) also loses less accuracy than a comparably-sized GGUF quant (Q5_K_M) — better quality and faster on this specific GPU.
