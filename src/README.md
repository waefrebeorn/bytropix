# `src/` — Core C Implementation

**13 files, ~10,000+ lines. Zero external ML dependencies.**

| File | Lines | Purpose |
|------|-------|---------|
| `wubu_model.c` | 1,343 | Model init, forward loop, MTP head, KV cache management |
| `wubu_ssm.c` | 2,741 | SSM Gated DeltaNet + GQA attention + AVX2 selective scan |
| `wubu_moe.c` | 555 | MoE router, quantized expert forward, shared expert |
| `wubu_mobius.c` | — | Möbius / Poincaré hyperbolic operations (experimental, not wired) |
| `wubu_nested_ssm.c` | — | Nested SSM variants (experimental, not wired) |
| `wubu_poincare_gqa.c` | — | Poincaré GQA attention (experimental, not wired) |
| `quantized_matmul.c` | 397 | Q8_K activation quantization → vec_dot dispatch, fused Q8_K variant |
| `quantized_dot_generic.c` | 1,125 | All 7 quant type vec_dot implementations (AVX2/SSE/generic) |
| `gguf_reader.c` | 1,787 | GGUF v3 parser, dequantization, blob buffer management |
| `wubu_tokenizer.c` | 300 | GPT-2 BPE tokenizer (248K vocab, merge-table-based) |
| `cuda_kernels.cu` | 2,452 | GPU kernels: SSM recurrence, attention, MoE, output projection |
| `gpu_output_proj.cu` | 272 | GPU output projection (cuBLAS SGEMM + Q4_K custom kernel) |
| `gpu_ssm_recurrence.cu` | 147 | GPU SSM recurrence kernel + K-head repeat |
| `gpu_moe_kernel.cu` | 147 | GPU MoE expert kernel (IQ2_XXS) |
| `wubu_model_gpu.cu` | 1,170 | GPU forward orchestration: SSM full forward, GQA, MoE |

## Key Design Points

- **Self-hosted vec_dot**: All quant types in one file. No libggml-cpu.so.
- **Direct blob pointers**: Quantized weights accessed via pointer into GGUF buffer. No F32 dequant copy for large weights.
- **Q4_0 KV cache**: 4-bit quantized cache (4:1 vs F16). Defined in `include/wubu_model.h`.
- **Fused Q8_K quant**: One Q8_K quantization per token, shared across all projections.
- **Tiled GQA attention**: Read K cache once per KV head (2 reads/position) instead of per Q head (16 reads).
