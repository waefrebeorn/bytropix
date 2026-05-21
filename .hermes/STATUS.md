# bytropix — True State (May 21 PM v24 — Phase 29c)

## Ground Truth
**CPU inference WORKS with FORCE_CPU_SSM_SEQ=1.** Coherent text produced.
**CPU inference BROKEN without FORCE_CPU_SSM_SEQ=1** — chunked SSM CS>1 FP accumulation.
**1:1 parity NOT achieved** — hidden states diverge from L0 (cs=0.405), but final output token matches reference.

## What Works
- CPU gen_text_cpu: 3-4 tok/s decode, sequential SSM, coherent output ✅
- Q4_0 KV cache: 4:1 compression, 720MB at 256k ✅
- Architecture: 3:1 SSM/GQA interleaved pattern (GQA at layers 3,7,11,15,19,23,27,31,35,39) ✅
- All quant types: Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ3_XXS, IQ4_XS, Q8_0 ✅
- GPU vision encoder: 0.52s ViT, 15.7s full pipeline ✅
- MTP spec decode: 8.5 tok/s, 4% acceptance ✅
- GPU quant matmul: Q5_K, Q6_K, Q4_K — single+batched ✅
- GPU hybrid text: 5.5 tok/s, coherent ✅
- **DUMP_INTERMEDIATE_DIR**: Rebuilt in llama.cpp (libllama.so + llama-simple) ✅
- **DUMP_GQA_DEBUG_DIR**: Per-layer GQA intermediate dumps in bytropix ✅
- **DUMP_GQA_LAYER**: Targeted layer debug env var ✅
- **gen_text symlink**: gen_text → gen_text_cpu ✅

## What's Broken
- Chunked SSM CS>1: FP accumulation across 30 layers ❌
- GPU text net-negative: H2D/D2H overhead + thermal throttling ❌
- 1:1 parity: Hidden states diverge from L0 (cs=0.405) — root cause unknown ❌
- gen_text_gpu: Links but untested this session ❓
- LLM claims corrected: L31 GQA (cs=0.471) is NOT the primary divergence source. SSM layers accumulate error faster.
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
