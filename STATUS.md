# STATUS — bytropix Inference Engine (May 19 PM, Phase 22)

**Overall cos-sim: 0.9994 vs llama.cpp** | CPU decode: ~9 tok/s | VRAM at 256k: ~6.45 GB

## What Works (✅ Verified)
- CPU `gen_text`: full 40-layer inference, ~11 tok/s prefill, Q4_0 KV cache
- All 7 quant types: Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ3_XXS, IQ4_XS, Q8_0
- `ref_dumper`: multi-token prompt support, DUMP_INTERMEDIATE_DIR (53 tensor types/layer)
- Architecture: 3:1 SSM/GQA interleaved (discovered May 19 — GGUF tensor enum)
- Q4_0 KV cache: 720 MB at 256k, 4:1 compression, cos-sim 0.9994 vs F16
- GPU pipeline: output proj, GQA attention, SSM recurrence, MoE experts
- L00-L30 per-layer cos-sim: 0.998-0.9999
- 13 bugs fixed (see README.md for full history)

## What's Broken (❌)
- `gen_text_gpu`: pre-existing hang after model load (unrelated to Phase 22)
- MTP speculative decode verify: 100% rejection at IQ2_M
- L31 cos-sim: 0.9585 (quantization noise through 30 layers — expected but monitored)

## Next (P0)
1. Debug gen_text_gpu hang
2. GPU Q4_0 KV cache (saves 3.7 GB VRAM at 256k)

## Build
```bash
make gen_text      # CPU inference
make ref_dumper    # Reference comparison
```

## Quick Verify
```bash
DUMP_LAYER_DIR=/tmp/ref ./ref_dumper /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "Your prompt" 0
DUMP_LAYER_DIR=/tmp/our ./gen_text "Your prompt" 0
tools/layer_cos_sim /tmp/ref /tmp/our 40
```
