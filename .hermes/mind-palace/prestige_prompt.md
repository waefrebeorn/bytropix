# Prestige Prompt — May 18, 2026

## Project: bytropix — Qwen3.6-35B-A3B-UD-IQ2_M

### Architecture (qwen35moe → qwen3next.cpp)
40 layers: 30 SSM (Gated DeltaNet) + 10 GQA (full attention)
- L0,1,2 → L3 GQA → L4,5,6 → L7 GQA → ... → L39 GQA
- Hidden: 2048, Vocab: 248320, Expert dim: 512, Shared dim: 512

### REAL STATUS (not markdown fantasy)
- F32 fallback (no quantization anywhere): cos-sim -0.128
- No MoE: cos-sim -0.157
- F32 MoE: cos-sim -0.51
- Quantized MoE (just wired): cos-sim -0.65
- **Root cause is SSM/GQA forward architecture, NOT quantization**

### What Was Actually Verified (unit tests only)
- Q4_K vec_dot vs F32 SGEMM: cos-sim 0.99995 ✅
- Q5_K vec_dot vs F32 SGEMM: cos-sim 0.99996 ✅
- Q6_K vec_dot vs F32 SGEMM: cos-sim 0.99996 (unit test only, NOT vs llama.cpp)
- IQ2_XXS/IQ3_XXS/IQ4_XS vec_dot: cos-sim 0.9999+ vs F32 SGEMM (unit test)
- Output projection: Q4_K quantized vs F32 SGEMM: cos-sim 0.99995 ✅

### What Was Wired This Session
- MoE quantized path: router(F32 blob) + shared expert(Q5_K/Q6_K blob) + routed experts(IQ2_XXS/IQ3_XXS/IQ4_XS blob) all through quantized_matmul
- load_from_blob flag to prevent double-free

### REAL NEXT: Per-layer reference dump → find first divergence → fix SSM/GQA
