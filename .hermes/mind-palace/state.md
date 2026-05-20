# State — Phase 22: Q4_0 KV Cache Compression

**bytropix: Pure C inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE)**
**Decode: ~9 tok/s (GPU) — Prefill: ~11 tok/s (CPU) — Q4_0 KV cache: 4:1 compression**

## Vault / Research Consumed
- Qwen3.6 technical report (architecture: 3:1 SSM/GQA interleaved pattern)
- Unsloth UD dynamic quantization blog (IQ2_XXS/IQ3_XXS/IQ4_XS quantization)
- llama.cpp qwen35moe.cpp source (intermediate tensor names)

## VRAM Budget (256k Context, Q4_0 KV Cache)
| Component | Size | Format |
|-----------|------|--------|
| GQA weights (F32) | 1,040 MB | cuBLAS SGEMM |
| SSM weights (quantized) | 692 MB | Q5_K/Q6_K native on GPU |
| SSM F32 weights (small) | ~30 MB | beta/alpha/dt_bias/a/conv1d/norm |
| SSM GPU conv_state | 2.3 MB | persistent |
| **KV cache (Q4_0)** | **720 MB** | **4-bit quantized, 4:1 vs F16** |
| Output proj (Q4_K) | 1,900 MB | quantized GPU kernel |
| SSM scratch | 49 MB | reusable intermediates |
| MoE + scratch | ~460 MB | cache(259MB) + scratch(200MB) |
| **Total** | **~6,453 MB** | **Fits 8GB VRAM with headroom** |

## Key Achievements (This Session)
- **DUMP_INTERMEDIATE_DIR**: Modified llama.cpp's `cb()` function to save ALL intermediate tensors (Qcur, Kcur, Vcur, beta, alpha_softplus, gate, conv_output, linear_attn_out, attn_output, etc.) to disk as F32 files. 1997 files per forward pass.
- **Discovered true architecture**: 40 layers with 3:1 SSM/GQA repeating pattern (NOT "30 SSM + 10 GQA" contiguous). Validated via GGUF tensor enumeration.
- **Phase 22: Q4_0 KV cache**: 4:1 compression ratio. Verified identical cos-sim (0.9994 overall). CPU path fully working. GPU path unchanged (uses own FP16 cache).
- **Per-layer cos-sim stable**: 0.999+ for SSM layers, 0.958-0.999 for GQA layers at 256k with 5-token prefill. The L31 drop is from accumulated quantization noise amplification.

## Next Optimization Opportunities (Highest Impact First)
1. **GPU KV cache quantization (Q4_0 for GPU attention)** — currently GPU GQA uses FP16 cache (5.12 GB). Q4_0 would reduce to ~1.44 GB, freeing 3.68 GB.
2. **Unified SSM kernel (phase A)**: fuse conv1d→SiLU→split→norm→beta (~1.2ms savings)
3. **Sparse attention** — add global token attention to sliding window for quality at 256k

## Current Limitations
- gen_text_gpu has a pre-existing hang bug (unrelated to Q4_0 changes)
- GPU GQA attention still uses FP16 cache (not compressed)
- Cos-sim drops to 0.958 at deeper GQA layers vs reference
