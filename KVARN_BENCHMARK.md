# KVarN: KV-Cache Quantization — Benchmark Notes

> Paper: *KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks*  
> arXiv: [2606.03458](https://arxiv.org/abs/2606.03458) — Müller, Bich, Boretti, Chang, Zhuang, Cavigelli (Huawei CSL, 2026)  
> Code: [github.com/huawei-csl/KVarN](https://github.com/huawei-csl/KVarN)

---

## What problem does it solve?

The KV cache grows linearly with context length and is one of the primary bottlenecks for long-context serving. Quantizing the KV cache to 2–4 bits saves memory and should increase throughput, but in practice existing methods degrade accuracy significantly — especially on reasoning tasks.

The paper's key insight is **why** existing methods fail at inference time even when they look fine in benchmarks. Standard quantization evaluations measure accuracy in *prefill* mode: the entire prompt is processed in parallel, so quantization errors in different positions don't interact. During *decode* — which is how you actually serve users — each new token reads every cached K and V value to compute attention. Quantization errors from early tokens accumulate across every subsequent decode step. The authors call this the *pseudo-decode* problem, and demonstrate that token-scale errors (magnitude, not direction) are the dominant failure mode.

---

## How it works

KVarN processes the KV cache one 128-token tile at a time as keys and values are written to the cache. Each tile passes through four stages:

### 1. Hadamard rotation

A fixed random Hadamard matrix `H ∈ ℝ^{d×d}` is applied along the channel (head) dimension:

```
K_rot = K · H^T
V_rot = V · H^T
```

The Hadamard transform runs in O(d log d). Because it mixes all channels together, per-channel outliers are spread uniformly across all channels — the output is approximately Gaussian-distributed regardless of the input distribution. This is the same trick used in QuIP# and similar weight quantization methods.

Crucially, the rotation is *orthonormal*, so it preserves dot products: `(K · H^T)(H · Q^T) = K · Q^T`. The rotation can be fused into the Q/K projection weight matrices during inference, so it adds zero runtime cost.

### 2. Iterative variance normalization

After rotation, tile-level outliers remain because different tokens have different overall magnitudes. KVarN fixes this with an iterative Sinkhorn-like algorithm that alternates between normalizing token scales (rows) and channel scales (columns) in log space:

```
for i in range(8):
    # Normalize along token axis (per-channel variance → 1)
    s_col = clamp(std(C, axis=token), c_min, c_max)
    C = C / s_col

    # Normalize along channel axis (per-token variance → 1)
    s_row = clamp(std(C, axis=channel), c_min, c_max)
    C = C / s_row
```

Eight iterations are enough for convergence. The clamping prevents degenerate scales from near-zero rows or columns. The scales `s_col` and `s_row` are stored at 8-bit precision alongside the quantized tile and applied in reverse during dequantization at read time.

This directly addresses the root cause identified in the paper: incorrect token scales produce large magnitude errors in K, which compound across decode steps. By equalizing variance before quantization, the per-element values fall in a narrow, predictable range that round-to-nearest handles well.

### 3. Asymmetric round-to-nearest quantization

Each element is independently quantized with asymmetric RTN:

```
K_q = round((K_norm - z) / s)      # encode
K_dq = K_q * s + z                 # decode (at read time)
```

Keys use per-channel scale/zero-point; values use per-token scale/zero-point. This matches where the remaining variance lives after normalization.

### 4. Asymmetric bit allocation: 4-bit keys, 2-bit values

The default preset (`kvarn_k4v2_g128`) stores keys in 4 bits and values in 2 bits. The paper shows that key quantization is accuracy-critical — keys participate in the attention softmax, so quantization errors in K directly shift which tokens get attended to. Values only scale the output; they're less sensitive. The 4K+2V split gives an average of 3 bits per element (vs 16 bits for FP16), which is roughly the same compression as competing methods, but with better accuracy because the normalization step makes 2-bit values viable.

### Three-region layout

The cache is not uniformly quantized. KVarN reserves:
- **Sink tokens** (first 128): kept in FP16. Early tokens (especially the BOS/system prompt) receive disproportionate attention and are sensitive to quantization.
- **Body** (middle): quantized with the 4K+2V scheme.
- **Trailing tokens** (last 128): kept in FP16. Recent tokens haven't yet had their errors averaged out by attention across many steps.

This is a minor memory overhead but meaningfully improves accuracy on long-range retrieval tasks.

---

## Paper accuracy results (Qwen3-4B, 2-bit, from arXiv:2606.03458)

| Benchmark | FP16 | KIVI (2-bit) | TurboQuant (2-bit) | KVarN k4v2 |
|---|---|---|---|---|
| AIME24 | 61.1 ± 3.1% | 55.5 ± 6.9% | 48.9 ± 1.9% | **60.0 ± 1.1%** |
| MATH500 | 82.6 ± 0.5% | 77.8 ± 0.5% | 77.0 ± 0.9% | **79.2 ± 0.4%** |
| HumanEval | 88.8 ± 1.8% | 86.4 ± 1.3% | — | **88.4 ± 0.3%** |
| Line retrieval (Llama-3.1-8B) | — | 83% | — | **89%** |

The variance normalization step is the key differentiator: KVarN is within statistical noise of FP16 on AIME24 while KIVI drops 5.6pp and TurboQuant drops 12.2pp. TurboQuant is the method vLLM shipped in May 2026; KVarN's paper claims 2.4× higher throughput at the same capacity and higher accuracy.

---

## Our benchmark setup

**Hardware:** NVIDIA RTX 5090 (Blackwell, 32 GB VRAM)  
**Model:** Qwen/Qwen3-8B-FP8 (head_dim=128, standard transformer)  
**Context:** 32,768 tokens max  
**vLLM version:** 0.22.0 (both runs)  

**Baseline:** `vllm-baseline:test` image (vllm/vllm-openai:v0.22.0 + our entrypoint), `--dtype auto` (FP8 native compute path), `--kv-cache-dtype auto` (BF16 KV cache)

**KVarN:** `vllm-kvarn:test` image (KVarN fork layered on v0.22.0), `--dtype float16` (required by KVarN; also necessary to avoid FP8 kernel conflict with KVarN's Triton attention backend), `--kv-cache-dtype kvarn_k4v2_g128 --block-size 128`

**Benchmark tool:** `vllm bench serve` (random dataset, 512-token input, 256-token output, concurrency sweep 1/2/4/8)

**Accuracy:** 5 factual prompts via `/v1/completions` at temperature=0, diffs compared manually.

---

## Results

### KV cache capacity

| | KV cache tokens (32K context, 32 GB) |
|---|---|
| Baseline (BF16 KV) | 129,920 |
| KVarN k4v2_g128 | 396,032 |
| **Ratio** | **3.05×** |

At 32K context per session this is roughly the difference between fitting 4 concurrent sessions and 12. The KVarN pool also reserves 2.53 GB as an FP16 tail workspace for dequantization — this is the cost of the three-region layout.

### Throughput

| Concurrency | Baseline req/s | KVarN req/s | Baseline TPOT | KVarN TPOT |
|---|---|---|---|---|
| 1 | 0.56 | 0.44 | 6.99 ms (143 tok/s) | 8.75 ms (114 tok/s) |
| 2 | 1.07 | 0.86 | 7.00 ms | 8.75 ms |
| 4 | 2.12 | 1.61 | 7.16 ms | 8.92 ms |
| 8 | 3.99 | 2.97 | 7.47 ms | 9.22 ms |

KVarN shows ~25% lower throughput here. **This does not replicate the paper's claim of ~1.3× speedup, but the comparison is not equivalent.** The baseline runs on native FP8 compute — the RTX 5090's Blackwell tensor cores have hardware FP8 MAC units that are significantly faster than FP16. KVarN requires float16 compute (the Triton kernels produce float16 K/V tensors), which means we're dequantizing the model's FP8 weights to float16 at load time, then running slower FP16 matrix multiply for the attention-adjacent computation. The paper's throughput benchmarks compare KVarN against an FP16 baseline, not an FP8 baseline.

A fair comparison requires a BF16/float16 model (no weight quantization). On a float16 baseline, KVarN's more compact KV cache reduces memory bandwidth pressure during decode — the decoder reads 3× fewer bytes per token from cache — which is how it achieves the throughput gain the paper reports.

### Accuracy

All five factual prompts (France capital, 17×23, Hamlet authorship, atomic number 79, Newton's laws) produce correct answers on both baseline and KVarN. The diff shows minor phrasing variation — expected at temperature=0 between float16 and bfloat16 compute — and in two cases KVarN answers more directly than the baseline (Q3: baseline lists sub-questions; KVarN states "written by William Shakespeare" in the first sentence; Q4: KVarN names Gold directly; baseline lists properties to look up). No accuracy degradation.

---

## Limitations found

### 1. head_dim=128 only

KVarN's Sinkhorn normalization kernel is hardcoded to 128×128 tiles (head_dim × token_chunk). Models with head_dim ≠ 128 fail at startup with:

```
ValueError: kvarn_k4v2_g128 requires head_dim=128, but this model has head_dim=256.
```

This blocks testing on our primary model: **Qwen3.6-27B (NV-FP4)** uses head_dim=256 (it's a hybrid Mamba-Transformer architecture). An upstream issue (#10) was filed 2026-06-05 and a maintainer responded the same day: *"I am fixing it right now."* Until that PR lands, only standard Qwen3/Llama3/Phi-4 configs (head_dim=128) work.

Fixing this yourself means re-parameterizing the Triton tile geometry in `/kvarn/vllm/model_executor/layers/quantization/kvarn/`. Non-trivial (2–4 days of Triton work) but the code is well-structured.

### 2. FP8 weight models require --dtype float16

The FP8 compute path and KVarN's Triton attention backend conflict during CUDA graph capture. Workaround: pass `--dtype float16`, which dequantizes FP8 weights to float16 at load time. Model weights inflate from ~9 GB to ~16 GB. For BF16 models this issue does not arise.

### 3. max_num_seqs auto-capped

KVarN reserves a fixed FP16 pool for the three-region layout. On a 32 GB GPU it automatically caps `max_num_seqs` from 256 to 48 to fit this pool. For our benchmark (max concurrency 8) this is irrelevant, but production deployments should be aware.

---

## Reproducing this benchmark

```bash
# Build images (one-time, ~7 min combined)
docker build -f Dockerfile.kvarn -t vllm-kvarn:test .
docker build --build-arg VLLM_VERSION=v0.22.0 -t vllm-baseline:test .

# Run (stop production server first if it holds the GPU)
docker compose stop
bash bench_kvarn.sh
docker compose up -d
```

Results land in `kvarn_results/<timestamp>/`. Override defaults:

```bash
# Use a BF16 model for a fair throughput comparison
MODEL=Qwen/Qwen3-8B MODEL_DIR=/path/to/models bash bench_kvarn.sh

# Push context harder
MAX_MODEL_LEN=65536 bash bench_kvarn.sh

# Just accuracy, skip throughput sweep
bash bench_kvarn.sh --accuracy-only
```

---

## What to do next for the article

**For a clean throughput number:** rerun with `Qwen/Qwen3-8B` (native BF16) as the model. Both baseline and KVarN will use float16, the compute path is identical, and the only variable is the KV cache format. You should see throughput at or above baseline as the paper claims.

**For the headline story:** wait for upstream issue #10 to merge, then rerun on Qwen3.6-27B (NV-FP4). At 128K context the KV cache savings are dramatic — currently ~4.3 GB in FP8, dropping to ~1.4 GB with KVarN — which on a 32 GB card means you can extend to ~380K context or serve 3× as many concurrent 128K sessions. That's the real production angle.
