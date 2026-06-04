# Qwen3.6-27B vLLM Setup — RTX 5090

## Why vLLM over llama.cpp for this hardware

The RTX 5090 (Blackwell, sm_120) has dedicated FP4 tensor cores. vLLM with NV-FP4
uses these natively via NVIDIA modelopt; llama.cpp runs standard CUDA kernels and
cannot use them yet. Expected throughput difference: ~130-150 t/s (llama.cpp Q5_K_M
with MTP) vs ~200-300 t/s (vLLM NV-FP4 with MTP).

## Why NV-FP4 beats Q5_K_M in quality despite being fewer bits

NV-FP4 uses FAAR (Format-Aware Adaptive Rounding) from NVIDIA modelopt. This
calibrates rounding per-weight using representative data, minimising error on
the most sensitive weights. Result: ~0.8% perplexity vs BF16, vs ~1.5-2% for
Q5_K_M. Higher quality AND faster on Blackwell.

## Model choice

**Primary:** `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`
- Text-only (no vision encoder overhead)
- MTP tensors confirmed present — speculative decoding works
- Already downloaded to `llm_models/hf/`

**Candidate:** `unsloth/Qwen3.6-27B-NVFP4` (downloading in background)
- Better calibration (Unsloth Dynamic 2.0 per-layer sensitivity)
- No "MTP" in name — MTP support unconfirmed; test before enabling spec-decode
- To switch: set `MODEL_NAME=unsloth/Qwen3.6-27B-NVFP4` in `.env`
  and comment out the `--speculative-config` line in `vllm.args` if MTP fails

## Key config decisions

### `ENFORCE_EAGER=` (disabled)
CUDA graphs pre-compile the forward pass and eliminate Python overhead between
batches. On Blackwell NV-FP4 this gives ~30-40% throughput improvement. The
original config had `ENFORCE_EAGER=1` which disables them — likely a workaround
for an older vLLM bug, no longer needed on current vLLM.

### `MAX_MODEL_LEN=131072` (128K)
Increased from 80K. With NV-FP4 weights at ~13.5 GB and FP8 KV cache, the full
128K context only requires ~4.3 GB of KV cache on a 27B model (GQA with 4 KV
heads means very compact KV). Total at 128K: ~18-19 GB, comfortably within 32 GB.

### `GPU_MEMORY_UTILIZATION=0.90`
Reduced from 0.95 to give vLLM room to allocate the larger KV cache pool for
128K without hitting OOM during the KV cache block pre-allocation step.

### `KV_CACHE_DTYPE=fp8_e4m3`
FP8 KV cache halves memory vs FP16. The e4m3 variant (4 exponent bits, 3 mantissa)
is the correct format for the RTX 5090 hardware FP8 path.

### `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'`
MTP (Multi-Token Prediction) uses a draft head baked into the model weights to
speculatively generate 3 tokens per step. The verifier accepts/rejects in parallel.
- `num_speculative_tokens: 3` — optimal for single-user throughput on this GPU
- `method: qwen3_5_mtp` — correct vLLM method name as of vLLM 0.19+ (old: `mtp`)
- Expected acceptance rate: ~87% / 72% / 61% for tokens 1/2/3 → mean ~3.0 accepted

### `--enable-prefix-caching`
Reuses computed KV blocks when multiple requests share a common prefix (e.g. same
system prompt). Essentially free win for any repeated-context usage.

### `--quantization modelopt`
Tells vLLM this is a modelopt-format NVFP4 checkpoint and to use the fast native
Blackwell FP4 kernel path rather than compressed-tensors fallback. The fallback
is ~1.74× slower on Blackwell according to benchmarks.

## VRAM breakdown at 128K context

| Component | Size |
|---|---|
| NV-FP4 model weights | ~13.5 GB |
| FP8 KV cache (128K, 4 KV heads) | ~4.3 GB |
| CUDA graphs + activations | ~1-2 GB |
| **Total** | **~19-20 GB** |
| Available (0.90 × 32 GB) | **28.8 GB** |
| Headroom | **~9 GB** |

This headroom allows for 256K context if needed (`MAX_MODEL_LEN=262144`), which
would add ~8.6 GB of KV cache and still fit.

## Switching models

```bash
# Switch to unsloth (once downloaded — check status with ./download_model.py --status)
# 1. Edit .env: MODEL_NAME=unsloth/Qwen3.6-27B-NVFP4
# 2. Comment out --speculative-config in vllm.args (unconfirmed MTP support)
# 3. Restart: ./run_vllm_server.sh

# Switch back to sakamakismile
# 1. Edit .env: MODEL_NAME=sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP
# 2. Uncomment --speculative-config in vllm.args
# 3. Restart: ./run_vllm_server.sh
```

## Why not llama.cpp anymore (for this GPU)

llama.cpp is still the right tool when:
- Running on Vulkan (AMD/Intel iGPU) — no NV-FP4 equivalent
- Needing GGUF model format
- Running without NVIDIA container toolkit

For the RTX 5090 specifically, vLLM NV-FP4 dominates on every metric that matters.
The llm-stack is kept for fallback and non-Blackwell deployments.
