# Plan — May 19, 2026 Late PM (Phase 14: SSM AVX2 Optimization ✅)

## Phase 0-13: DONE ✅
| Phase | Detail | Status |
|-------|--------|--------|
| 0-11 | GQA attn, AVX2 vec_dot, self-contained, MoE, quant path, KV cache | ✅ |
| 12 | MTP Speculative Decode | ✅ MTP broken at UD-IQ2_M |
| 13 | GPU Output Projection (cuBLAS SGEMM, batched prefill) | ✅ |

## Phase 14: SSM AVX2 Optimization — DONE ✅
### Fused Q8_K Quantization
- SSM: attn_qkv (Q5_K, 8192 cols) + attn_gate (Q5_K, 4096 cols) share Q8_K quant
- GQA: Q+gate (Q5_K, 8192) + K (Q5_K, 512) + V (Q5_K, 512) share Q8_K quant  
- New `quantized_matmul_from_q8()` function takes pre-quantized Q8_K buffer
- Saves 50 Q8_K quantize operations per decode step

### AVX2 Selective Scan Inner Loops
Four helper functions with AVX2 intrinsics added to wubu_ssm.c:
1. `avx2_state_decay()` — `_mm256_mul_ps` over 16384 elements
2. `avx2_hk()` — `_mm256_fmadd_ps` + HSUM256 reduction
3. `avx2_state_update()` — `_mm256_fmadd_ps` outer product
4. `avx2_hq()` — `_mm256_fmadd_ps` + HSUM256 reduction
Each processes 8 floats/iteration vs scalar's 1. Verified: max diff 1.85e-3.

### GPU Quantized Output Projection (GPU_QUANTIZED=1)
- Custom CUDA kernel: one thread per vocab column, dequants Q4_K on-the-fly
- Uses ~1.9GB VRAM vs F32 mode's 7.6GB
- Makes GPU output proj viable on 6.5GB VRAM laptops
- Falls back to CPU if GPU memory insufficient

### NaN Guard Optimization
- GQA NaN/Inf check gated behind DUMP_GQA_DEBUG env var
- Saves ~90K isnan() calls per decode

## Phase 15: 256k Context Optimization [P1 — NEXT]
GQA attention is O(n) per layer at decode time (attending over cache_len positions).
At 256k context: 2 KV heads × 256k × 256-dim dot ≈ 131M FMA per GQA layer.
Total: 10 GQA layers × 131M = 1.3B FMA → ~13ms at 100 GFLOPS.

### Options:
1. **Flash attention**: Tile QK^T computation to reduce memory bandwidth
2. **KV cache tiering**: Keep most tokens in F16 on GPU, only attend to recent tokens in CPU
3. **Streaming attention**: Window-based attention with compressed historical context
4. **GPU GQA attention**: Port the Q@K^T dot product and V weighted sum to CUDA

## Phase 16: MTP Speculative Decode (Unblocked) [P2]
Requires UD-Q2_K_XL model. Our two-model load, DRAFT_N=2, checkpoint/rollback is correct.

## Phase 17: MoE Router Prefetch + Expert Caching [P3]
- Expert indices from previous layer prefetch next layer's weights
- Already partially implemented (stride prefetch to L3)
- Could cache frequently-used experts' dequant values
