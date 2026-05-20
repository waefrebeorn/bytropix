# STATUS — bytropix Inference Engine (May 21 PM, Phase 28b DA)

**GPU decode (SSM on CPU): 7.6-8.5 tok/s (4K ctx)** | **GPU_SUPPORT now live — SSM GPU path UNVERIFIED**

## What Works (✅ Verified at runtime)
- GPU `gen_text_gpu`: full 40-layer, no hang, Q4_0 KV cache default
- Q4_0 fused decode attention: 8.1 tok/s (beats FP16 7.6)
- Fused Q5_K/Q6_K quant matmul (row_major): compiles and runs
- **GPU_SUPPORT builds and runs**: wubu_model_gpu_ssm_forward_full() called per layer

## DA-Corrected Status (❓ = no longer trust old claims)
- ❓ SSM GPU path correctness: **NEVER verified vs CPU path**. All cos-sim claims were from isolated tests or dead code comparisons
- ❓ 256k output cos-sim vs llama.cpp (only verified at small context with dead GPU_SUPPORT)
- 🔴 F32 dequant SSM weights: ~2.2 GB wasted VRAM, never used in inference
- 🔴 GPU memory leak: quantized + F32 SSM weights never freed
- 🔴 Prefill N>1 fallback uses broken column-major kernel

## P0 Fixes (Phase 28b)
- Remove F32 dequant weight upload (save 2.2 GB)
- Fix wubu_model_gpu_free() memory leak
- Fix prefill N>1 fallback kernel (use row_major)
- Fix gen_text.c prompt for proper testing
- Cos-sim: GPU SSM vs CPU SSM

## Build
```bash
make gen_text      # CPU inference
make gen_text_gpu  # GPU inference (GPU=1 env var for GPU init)
```

