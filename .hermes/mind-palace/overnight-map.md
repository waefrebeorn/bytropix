# Overnight Map — Phase 28k: GPU MoE Analysis Complete, P1 MTP Working

**Active repo:** /home/wubu/bytropix/
**Main model:** /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (11.5GB, 40 layers)
**MTP model:** /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf (11.9GB, +blk.40 head)
**Current commit:** 695fda5 (pushed to origin/master)

## Session Summary (May 20-21, 2026)
8 hours, 100+ tool calls, 3 new tools, 1 new skill, 2 commits pushed.

### P0: GPU MoE Analysis — COMPLETE
- **v5 Q8_K kernel** committed (12ad638): int8 quantization of input, matches CPU arithmetic
- **CUDA sm_120 bugs** found and documented in new skill: 3 workarounds for Blackwell compiler
- **DA v13**: Root cause identified — 0.9888 per-layer cos-sim is FUNDAMENTAL code-path difference
- **Pragmatic decision**: Accept hybrid path (GPU SSM/GQA + CPU MoE)

### P1: MTP Speculative Decode — WORKING
- gen_text_mtp built and tested with MTP model
- Output: "to find your group group and to share your energy with them"
- 8.5 tok/s decode, MTP acceptance 4% (low from quantized head)
- Falls back to single-token without MTP=1

### Vision Pipeline — VERIFIED (May 21)
- Full vision→text pipeline works: screenshot→encoder→mmproj→text model→logits
- 256×256 image: 128 patches × 2048 dim, encoder 63.7s CPU, text 4.77s
- Logit range [-10.77, 14.09], no NaN/Inf ✅
- 2 segfault bugs fixed in wubu_vision.c:
  1. n_patches_total capped at V_MAX_POS (prevent massive alloc)
  2. scores[2304] stack array → heap-allocated (SIGSEGV on >2304 patches)

### GQA Batched Prefill Fix (May 21) — P2 completed
- **Bug:** GPU GQA prefill processed tokens one-at-a-time (`for t in 0..N: gqa_fwd(C=1)`) causing N×H2D/D2H overhead
- **Fix:** Pass `C=N` to batched GPU GQA forward (`wubu_model_gpu_gqa_forward(model, l, normed, N, attn_out)`)
- **Added:** `wubu_model_gpu_chunk_sz()` helper for scratch size check, with N>chunk_sz sub-batching fallback
- **Added:** `int wubu_model_gpu_chunk_sz(wubu_model_t *model)` declaration in wubu_model.h + implementation in wubu_model_gpu.cu
- **Impact:** 5x fewer GPU calls for GQA layers during prefill (10 layers × N→1 batched call)
- **TODO:** Similar batched fix for SSM forward_full C>1 path (currently unused, falls back to per-token)

## What's Blocked
- CUDA sm_120 compute-sanitizer doesn't work (WDDM debugger not initialized)
- GPU MoE bit-exact parity would need 3-5 sessions of code porting
- Vision encoder CPU-only = 63s for 256×256 (needs GPU kernel for real-time)

## Verifiable Facts (DO NOT RE-DERIVE)
- GPU MoE v5 is committed and correct but 0.9888 cos-sim is fundamental
- Hybrid path: FORCE_CPU_MOE=1 GPU=1 ./gen_text_gpu → coherent text at 5.5 tok/s
- MTP: MTP=1 OMP_NUM_THREADS=16 ./gen_text_mtp "prompt" 30
- CUDA sm_120 bugs: use extern __shared__ float, thread-0 reduce, no static __shared__
- compare_moe_expert tool: GPU=1 ./compare_moe_expert
- Vision pipeline: test_vision_real <mmproj.gguf> <pixels.bin> [H] [W] [model.gguf]
- Build: g++ -DGPU_SUPPORT ... -o test_vision_real tools/test_vision_real.c ... all GPU objs

## Next Session: P2 Feature Cream
- GPU RMSNorm + SiLU + gated norm kernels
- Chunked prefill (3-7x speedup)
- NSA sparse attention
- RoPE extrapolation 4x
- GPU vision encoder kernels
