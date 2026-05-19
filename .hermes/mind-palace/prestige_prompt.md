# Prestige Prompt — May 19, 2026 Late PM (Phase 14 — SSM AVX2 Optimization ✅)

## Project: bytropix — Qwen3.6-35B-A3B-UD-IQ2_M
**Cos-sim: 0.9967** — 1:1 PARITY ACHIEVED!
**Decode: 8.8 tok/s CPU** — Phase 14 optimizations shipping.

## Phase 14: SSM AVX2 Optimization ✅

### The Four AVX2 Inner Loops
The SSM selective scan operates on a [128, 128] = 16384-element state matrix per head.
Before: 4 nested scalar loops (128×128 = 16384 iterations each).
After: AVX2 intrinsics processing 8 elements per instruction.

1. **State decay** (`avx2_state_decay`): `_mm256_mul_ps` — 2048 iterations
2. **h @ k** (`avx2_hk`): `_mm256_fmadd_ps` + horizontal reduction — 128 rows × 16 iterations
3. **State update** (`avx2_state_update`): `_mm256_fmadd_ps` outer product — 128 rows × 16 iterations
4. **h @ q** (`avx2_hq`): identical to h@k

### Fused Q8_K Quant
The SSM and GQA layers both use the SAME input for multiple projections.
By quantizing input→Q8_K ONCE per token and reusing across all projections,
we eliminate redundant computation.

**Before**: attn_qkv(): quant_x→Q8_K, compute 8192 cols. attn_gate(): RE-QUANT, compute 4096 cols.
**After**: quant_x→Q8_K ONCE. attn_qkv: 8192 cols. attn_gate: 4096 cols. Zero redundant quant.

### NaN Guard
The GQA NaN guard was checking every element (Q: 8192, K: 512, V: 512) per token.
That's ~10K isnan() calls × 10 GQA layers = 100K per decode. Gated behind DUMP_GQA_DEBUG.
In normal operation: no NaN guard overhead.

### GPU Quantized Output Proj
New mode: `GPU_QUANTIZED=1`. Custom CUDA kernel reads Q4_K blocks from GPU memory,
dequants on-the-fly, and accumulates dot product. Uses ~1.9GB VRAM instead of 7.6GB.

## Next: Phase 15 — 256k Context Optimization
GQA attention O(n) per decode step. At 256k context, this becomes the dominant cost.
Flash attention tiling or streaming attention window needed.
