# Overnight Map — May 19, 2026 Late PM (Phase 14 Complete)

## BREAKTHROUGH: SSM AVX2 Optimization + Fused Q8_K + GPU Quantized Mode

### SSM AVX2 Selective Scan
Four inner loops over the 128×128 SSM state matrix now use AVX2 intrinsics.
Each loop processes 8 floats per instruction (8× scalar throughput for mul/add/FMA).
The 4 loops: state decay (scalar mul), h@k (matvec), state update (outer product), h@q (matvec).

### Fused Q8_K Quantization
Added `quantized_matmul_from_q8()` — a variant of `quantized_matmul()` that takes
a pre-quantized Q8_K buffer instead of re-quantizing the input.

Applied to:
- SSM forward (Euclidean): attn_qkv + attn_gate share quant → saves 30 quants/decode
- SSM forward (save): same optimization  
- SSM forward (Möbius): same optimization
- GQA forward: Q+gate + K + V share quant → saves 20 quants/decode

### GPU Quantized Output Projection
Custom CUDA kernel: `quantized_output_proj_kernel` — one thread per vocab column.
Each thread reads Q4_K blocks from GPU memory, dequants on-the-fly (144 bytes → 256 floats),
and accumulates the dot product with the input vector.

**VRAM comparison**: 1.9GB (Q4_K) vs 7.6GB (F32 cuBLAS).
**Set `GPU_QUANTIZED=1`** to enable quantized mode.

### NaN Guard Optimization
GQA's per-element isnan()/isinf() check is now gated behind `DUMP_GQA_DEBUG` env var.
Saves ~90K isnan() calls per decode in normal operation.

### Files Modified
- `src/wubu_ssm.c` — 4 AVX2 helpers + fused Q8_K in all 4 forward variants + NaN guard gating
- `src/quantized_matmul.c` — new `quantized_matmul_from_q8()` function
- `include/gguf_reader.h` — declaration for `quantized_matmul_from_q8()`
- `src/gpu_output_proj.cu` — full rewrite with quantized CUDA kernel
- `include/gpu_output_proj.h` — no interface changes (backward compatible)
- `.hermes/mind-palace/state.md` — updated
- `.hermes/mind-palace/goal-mantra.md` — updated
- `.hermes/mind-palace/plan.md` — updated with Phase 15 plan
- `.hermes/mind-palace/prestige_prompt.md` — updated

### Build Commands
```
make gen_text              # CPU (8.8 tok/s)
make gen_text_gpu          # GPU + quantized kernel
GPU_QUANTIZED=1 ./gen_text_gpu "Hello" 32  # Quantized GPU mode
GPU=1 ./gen_text_gpu "Hello" 32            # F32 GPU mode (needs 8GB+ VRAM)
```

### Next: Phase 15 — 256k Context Optimization
GQA attention is O(n) per decode step × 10 layers. At 256k context:
- Each GQA layer: 2 KV heads × 256k × 256-dim dot = 131M FMA
- 10 layers: 1.3B FMA total
- Options: flash attention, KV cache tiering, streaming attention, GPU attention kernel
