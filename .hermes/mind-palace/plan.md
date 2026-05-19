# Plan — May 19, 2026 Late PM (Phase 14: SSM AVX2 ✅ → GPU Roadmap)

## Phase 0-14: DONE ✅
| Phase | Detail | Status |
|-------|--------|--------|
| 0-11 | GQA attn, AVX2 vec_dot, self-contained, MoE, quant path, KV cache | ✅ |
| 12 | MTP Speculative Decode | ✅ EMA logit correction implemented |
| 13 | GPU Output Projection (cuBLAS SGEMM, batched prefill) | ✅ |
| 14 | SSM AVX2 Optimization (fused Q8, scan, NaN guard, tiled GQA) | ✅ |

## GPU Acceleration Roadmap

### Current GPU Status (Phase 14 end)
```
Output proj  [GPU: ✅ Q4_K quantized kernel + F32 cuBLAS]
SSM matmuls [CPU only — 30 layers × 3 Q5_K matmuls = ~30ms]
GQA attn    [CPU only — 10 layers × Q·K over KV cache]
MoE         [CPU only — 8 experts × IQ2_XXS/IQ3_XXS = ~48ms]
MTP         [CPU only — blk.40 draft + verify]
```

### Phase 15: GPU GQA Attention [P1 — NEXT]
**Why first:** GQA attention at 256k becomes O(n) per decode. On CPU at 256k: ~230ms.
On GPU: stream KV through GPU in batches, compute Q·K on-the-fly.

**Approach:** Upload Q (8KB) once per GQA layer. Stream KV cache entries through GPU in batches (4096 per launch). Compute scores on GPU. Download scores + V weighted sum.

**Implementation:** New `gpu_gqa_attention_kernel()` in gpu_output_proj.cu. Uses the existing F16 KV cache on CPU, streams through GPU in tiles. Each tile: upload KV tile, compute Q·K, accumulate V sum, download partial sums.

**Expected speedup at 256k:** ~40ms → ~5ms per decode step (8×).

### Phase 16: GPU SSM Matmuls [P2]
**Target:** SSM attn_qkv (Q5_K, 2048×8192) + attn_gate (Q5_K, 2048×4096) = ~20ms total.
**Approach:** Upload hidden state [2048], run Q5_K dequant+matmul on GPU via custom kernel (similar to GPU_QUANTIZED output proj kernel). Keep each layer's Q5_K weights on GPU (8MB per layer × 30 = 240MB total).

**Challenge:** 30 layers × 3 matmuls × 2 uploads/downloads = 180 PCIe transfers. At ~32GB/s, each 8KB upload + 32KB download = ~1.25µs. Total overhead: ~225µs. Worth doing if GPU compute time is <20ms vs CPU's 20ms.

### Phase 17: GPU MTP Pipeline [P3]
**MTP is a GPU gainz** — the draft head (blk.40) is one extra layer. After SSM+GQA+MoE are GPU-accelerated:
- Main model forward + MTP draft forward as parallel GPU streams
- Draft verification becomes batch: verify draft[1] by re-running just the SSM/GQA layers with the draft token
- Overlap draft generation with main model inference via CUDA streams

**Expected speedup:** 1.15-1.25x (MoE models, per blog) from spec decode alone. Combined with GPU acceleration of base model: potential 3-5x overall.

### Phase 18: GPU MoE Expert Compute [P4]
**Hardest port:** Dynamic expert routing (256 experts, 8 active per token). Each expert is IQ2_XXS gate/up + IQ3_XXS down. Need to:
- Upload hidden state
- Run router (tiny, stay CPU)
- Upload selected expert indices to GPU
- GPU loads 8 experts × 3 weight matrices from device memory
- Compute gate*up→silu→down
- Download result

**Keep all 256 experts × 3 weight matrices on GPU:** IQ2_XXS gate/up: 66 bytes/block × 8 blocks/expert × 256 experts × 2 = ~270KB × 2 = 540KB. IQ3_XXS down: 98 bytes/block × 2 blocks × 256 = ~50KB. Total: ~590KB for all 256 experts! That's tiny — fits in L2 cache on GPU.

## MTP Status (Post-Phase 14)
**Online Logit Correction EMA implemented.** The MTP head (blk.40, Q2_K/Q3_K) produces systematically biased logits vs main model (IQ2_XXS/IQ3_XXS). Our fix:
- Running EMA of per-logit difference: `correction[v] = 0.9*c[v] + 0.1*(main_logits[v] - mtp_logits[v])`
- Applied before sampling: `mtp_logits[v] += correction[v]`
- Converges within ~10 tokens
- Pure math — no requant needed

**Next MTP step:** After GPU acceleration of SSM+GQA+MoE, MTP becomes a batched pipeline. Draft generation and verification can overlap on separate CUDA streams.

## Unchanged Items
| Phase | Detail | Status |
|-------|--------|--------|
| IQ3_XXS AVX2 vec_dot | MoE down weights, generic C only | Future |
| SSM conv1d SIMD | Not a hot path currently | Future |
| KV cache tiering | KV_WINDOW env var for streaming attention | Phase 15 option |
| Chat template | Not applied, minor quality impact | Low priority |
