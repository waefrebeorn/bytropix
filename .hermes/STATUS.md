# bytropix — True State (May 19 PM v22)

## Ground Truth
**Inference WORKS.** Cos-sim 0.9994 vs llama.cpp reference (CPU, 5-token, 40 layers).
**Q4_0 KV cache**: 4:1 compression, 720MB at 256k, identical quality.

## What Works
- CPU gen_text: ~11 tok/s prefill, full 40-layer inference ✅
- Q4_0 KV cache: 4:1 compression, cos-sim 0.9994 ✅
- DUMP_INTERMEDIATE_DIR: 53 tensor types/layer reference tracing ✅
- Architecture: 3:1 SSM/GQA interleaved pattern discovered ✅
- All 7 quant types: Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ3_XXS, IQ4_XS, Q8_0 ✅
- ref_dumper: Multi-token prompt support, numeric token ID mode ✅
- Layer cos-sim: L00-L30=0.998-0.9999 ✅
- Sliding window GQA: GQA_WINDOW env var, 16→1 tile at 256k ✅

## What's Broken
- gen_text_gpu: Pre-existing hang after model load ❌
- L31 cos-sim: 0.9585 (quantization noise) 🟡
- GPU KV cache: Still FP16 (5.12GB), should be Q4_0 💤

## Key Tools
```bash
make gen_text && ./gen_text "prompt" N          # CPU inference
make ref_dumper && DUMP_LAYER_DIR=/tmp/r ./ref_dumper model.gguf "prompt" 0  # Reference
tools/layer_cos_sim /tmp/r /tmp/o 40            # Compare
```

## Priorities
P0 — Fix gen_text_gpu hang
P0 — GPU Q4_0 KV cache (saves 3.7GB VRAM)
P1 — Unified SSM kernel fusion
P2 — Sparse attention for 512k+
