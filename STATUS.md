# STATUS — bytropix Inference Engine (May 20 PM, Phase 25)

**GPU decode: 7.6-8.5 tok/s (4K ctx)** | **256k: 4.8 tok/s full, 5.7 tok/s sliding window** | **VRAM: ~3.56 GB**

## What Works (✅ Verified)
- GPU `gen_text_gpu`: full 40-layer, no hang, Q4_0 KV cache default
- Q4_0 fused decode attention: 8.1 tok/s (beats FP16 7.6)
- Fused Q5_K/Q6_K quant matmul: incremental dequant+dot, no local mem spill
- Fused SSM beta/alpha decode: replaces 2 cuBLAS calls + 4 element-wise kernels
- Sliding window attention: GQA_WINDOW env var

## What's Unverified (❓)
- ssm_beta_alpha_fused_decode correctness vs old cuBLAS path
- 256k output cos-sim vs llama.cpp (only verified at small context)
- Bottleneck profiling data (guesses, not nsight measurements)

## What's Pending (P0-P2)
- P0: Fuse conv1d+SiLU+split+L2 norm into single SSM decode kernel
- P1: MoE router on GPU (eliminate CPU hop)
- P1: Systematic nsight profiling of decode bottlenecks
- P2: Chunked prefill (3-7x at 256k from Qwen2.5-1M paper)

## Build
```bash
make gen_text      # CPU inference
make gen_text_gpu  # GPU inference
make ref_dumper    # Reference comparison
make clean         # Remove all binaries
```
