# WuBuText AI — State Dashboard (May 16 AM v9)

## Inference Engines

| Binary | Status | Performance | Notes |
|--------|--------|-------------|-------|
| `infer_text` | ✅ v2 | KV cache + lazy MoE + SSM carry | **Prefill 2 tok 17.3s → decode 4 tok 27.7s (MOE=1)** |
| `train_integrated` | ✅ | 11s/step, CE 21.6→18.4, 0 NaN | Per-expert dequant, GPU proj |
| `infer_moe_lazy` | ✅ | 37 tok/s, 0.35s dequant (9×) | Lazy dequant benchmark |
| `infer_unified` | ✅ | 40 layers in 1 binary | SSM→GQA→MoE chain |
| `test_kv_cache` | ✅ | max_diff=0.00 vs recompute | KV cache: 1GB/layer @ 256K |
| `test_256k` | ✅ | MoE router 65K verified | SSM O(T), GQA needs KV cache |
| `infer_vision_gpu` | ✅ | GPU: 99ms (128×128) | 27 GPU layers, 0 NaN |

## infer_text v2 — KV Cache + Lazy MoE

**Architecture:**
- **Phase 1 (Prefill):** Full forward over prompt tokens, populate GQA KV caches per-layer
- **Phase 2 (Decode):** Token-by-token, GQA uses KV cache (no O(T²) recompute), SSM carries state across steps
- **Lazy MoE:** Dequant router + shared expert once at startup. Per-decode-step: route → collect unique expert IDs → dequant only new experts when routing changes. Direct expert lookup (no 3GB temp arrays)

**KV Cache per GQA layer:**
- Stores post-RMSNorm K, raw V
- CPU backing store with GPU mirror
- 1 token step: Q projection → K/V projection → RMSNorm → append to cache → attend against all cached K/V
- Online softmax fallback for T > 64K

**Memory:**
- Router + shared expert dequant (once): ~11MB per layer × 40 = 440MB
- Per-layer cached experts: ~96MB per layer (8 experts × 3 tensors × 1M floats)
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
