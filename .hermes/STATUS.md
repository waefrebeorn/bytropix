# bytropix — True State (May 21 PM v23)

## Ground Truth
**CPU inference WORKS with FORCE_CPU_SSM_SEQ=1.** Coherent text produced.
**CPU inference BROKEN without FORCE_CPU_SSM_SEQ=1** — chunked SSM CS>1 FP accumulation.

## What Works
- CPU gen_text_cpu: 3-4 tok/s decode, sequential SSM, coherent output ✅
- Q4_0 KV cache: 4:1 compression, 720MB at 256k ✅
- DUMP_INTERMEDIATE_DIR: 53 tensor types/layer reference tracing ✅
- Architecture: 3:1 SSM/GQA interleaved pattern ✅
- All quant types: Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ3_XXS, IQ4_XS, Q8_0 ✅
- ref_dumper: Multi-token prompt support, numeric token ID mode ✅
- GPU vision encoder: 0.52s ViT, 15.7s full pipeline ✅
- MTP spec decode: 8.5 tok/s, 4% acceptance ✅
- GPU quant matmul: Q5_K, Q6_K, Q4_K, IQ1_M — all single+batched ✅
- GPU hybrid text: 5.5 tok/s, coherent ✅

## What's Broken
- Chunked SSM CS>1: FP accumulation across 30 layers ❌
- GPU net-negative: H2D/D2H overhead makes GPU hybrid slower than CPU ❌
- gen_text binary: doesn't exist (only gen_text_cpu / gen_text_gpu) ⚠️
- L31 cos-sim: 0.9585 (quantization noise amplification) 🟡
- MTP acceptance: 4% due to quantized IQ2_M head 🟡

## Key Commands
```bash
FORCE_CPU_SSM_SEQ=1 ./gen_text_cpu "prompt" 20   # CPU inference (coherent)
tools/layer_cos_sim /tmp/ref /tmp/our 40           # Per-layer comparison
tools/ref_dumper model.gguf "prompt" 0              # Reference dumps
```

## Priorities
P0 — Llama.cpp inline hooks for intermediate debugging
P0 — Fix chunked SSM CS>1 or accept sequential-only
P1 — L31 attention divergence investigation
P2 — GPU KV cache Q4_0 (saves 3.7GB VRAM)
P3 — Sparse attention, RoPE, MTP improvements
