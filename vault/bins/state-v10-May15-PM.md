# WuBuText AI — State Dashboard (May 15 PM v10)

## Inference Engines

| Binary | Status | Performance | Notes |
|--------|--------|-------------|-------|
| `infer_text_gpu v5` | ✅ Chunked prefill + KV cache | Prefill 22 tok/s, decode 245 tok/s | GPU fwd wired. KV cache persistent. No full recompute |
| `infer_text` v2 | ✅ | KV cache + lazy MoE + SSM carry | **Prefill 2 tok 17.3s → decode 4 tok 27.7s (MOE=1)** |
| `train_integrated` | ✅ | 11s/step, CE 21.6→18.4, 0 NaN | Per-expert dequant, GPU proj |
| `infer_moe_lazy` | ✅ | 37 tok/s, 0.35s dequant (9×) | Lazy dequant benchmark |
| `infer_unified` | ✅ | 40 layers in 1 binary | SSM→GQA→MoE chain |
| `test_kv_cache` | ✅ | max_diff=0.00 vs recompute | KV cache: 1GB/layer @ 256K |
| `test_256k` | ✅ | MoE router 65K verified | SSM O(T), GQA needs KV cache |
| `infer_vision_gpu` | ✅ | GPU: 99ms (128×128) | 27 GPU layers, 0 NaN |

## infer_text_gpu v5 — Chunked Prefill + Persistent KV Cache

**Architecture:**
- **Prefill (chunked):** Prompt split into CHUNK-size blocks. Each chunk processed through all layers. GQA: QKV proj → RMSNorm → RoPE → append to persistent cache → chunked_attn against cache. SSM: process chunk (state persists in d_ss/d_cs). MoE: per-chunk tokens.
- **Decode (incremental):** No full-sequence re-evaluation. Single token: embed → QKV proj → RMSNorm → RoPE → append to cache → attention against all cached → output. SSM state carries. MoE for 1 token.
- **KV cache:** Persistent GPU buffers per GQA layer [maxT, kv_dim]. Accumulates during prefill, appended during decode.
- **chunked_attn kernel:** cuBLAS SGEMM per Q-head against cached K/V. GQA handled via per-head loops with correct leading dimensions. Softmax via single-row kernel.

**Performance vs v4 (full recompute):**
| Metric | v4 | v5 | Speedup |
|--------|-----|-----|---------|
| Prefill 10 tok | 15 tok/s | 18 tok/s | 1.2× |
| Decode 5 tok | 85 tok/s | 245 tok/s | 2.9× |
| Prefill 66 tok | — | 22 tok/s | — |
| Decode (MOE=1) | CPU-bound | Not tested (slow dequant) | — |

**Memory:** GPU scratch sized for one chunk (CHUNK=256). Score scratch = CHUNK * n_q * maxT. At 256K: 256*16*262144*4 ≈ 4GB. Fits 6.4GB with model weights (~3GB GQA).

## Known Issues
- `cublasHandle_t` C type warning in infer_text_gpu (binary works)
- Score scratch grows O(C * n_q * maxT) — tile internally at 256K
- MOE=1 not validated in v5 (dequant timeout in setup)
- No long-context (>64K) stress test
- GQA KV cache: 2 × max_T × kv_dim × sizeof(float) per GQA layer

## GQA KV Cache — All 256K Context Tests

| Component | Scaling | Status |
|-----------|---------|--------|
| MoE Router | O(T) | ✅ Verified to 65K |
| SSM forward | O(T) | ✅ Verified via GPU kernel |
| GQA attention | O(T) with KV cache | ✅ Cached decode working |
| Token embeddings | O(T) | ✅ 2GB at 256K |
| Output projection | O(T×V) | ⚠️ 256K×248K×2K = impractical |
| Generation | O(T) per step (cache) | ⚠️ CPU bottleneck, no GPU forward acceleration |

## Known Issues

| Issue | Impact | Status |
|-------|--------|--------|
| ~7s/token decode (MOE=1) | Slow generation | CPU-bound: 40 layers × direct MoE |
| PGA loss jumps 21.6→69 | PGA backward LR too high | Needs LR scaling |
| CPU output projection ~2s/token | O(N×V×D) for V=248320 | ✅ Moved to GPU (0.5ms) |
| No GPU forward acceleration | All layers on CPU | Has GPU kernels (gpu_gqa_forward, gpu_ssm_forward) but not wired |

## TGT Math
BOUNDARY = 2π
remainder = fmod(x + π, BOUNDARY) - π
tgt_safe_expf(x) = x > 80 ? 80 : x < -80 ? 0 : expf(x)
