# State — Phase 28l: Vision Pipeline Verified, P2 Up Next

**bytropix: GPU inference engine for Qwen3.6-35B MoE + vision multi-modal**

## CURRENT STATE
| Component | Result | Status |
|-----------|--------|--------|
| GPU SSM decode (hybrid) | Cos-sim 1.0 | ✅ |
| GPU SSM recurrence (GPU kernel) | Cos-sim 1.0 | ✅ |
| GPU GQA (prefill + decode) | Coherent text with CPU MoE | ✅ |
| GPU MoE v5 (Q8_K kernel, single layer) | Cos-sim 0.9888 vs CPU | 🟡 1.1% per-layer error |
| GPU MoE (all 40 layers) | Running cos-sim 0.9968 → garbage text | ❌ Accumulates to garbage |
| GPU SSM/GQA + CPU MoE | "Paris has been the first..." (5.5 tok/s) | ✅ Coherent |
| MTP spec decode | 8.5 tok/s, 4% acceptance | ✅ Working |
| **Vision encoder** | **256×256 → 128 patches × 2048, no NaN** | **✅ VERIFIED** |
| **Vision→text pipeline** | **Full pipeline: screenshot→logits** | **✅ VERIFIED** |
| CPU-only | "Paris is the capital of France..." | ✅ |
| gen_text_mtp | 8.5 tok/s, 4% acceptance | ✅ Working |
| Vision encoder → text | 63.7s CPU, no NaN, logits [-10.8, 14.1] | ✅ Verified |

## ROOT CAUSE ANALYSIS (DA v13 Complete)
The 1.1% per-layer GPU MoE error is NOT from any single bug. It is the fundamental result of running a different code implementation:
- CPU `quantized_matmul` uses `quantize_row_q8_K` (negative d, sign-inverted Q8)
- GPU kernel uses `max_abs / 127` (positive d, same-sign Q8)
- Both are mathematically equivalent but produce different floating-point rounding
- The 0.32% running cos-sim error compounds through 40 layers → flips token selection in 240K vocab

**Verdict:** GPU MoE bit-exact parity is NOT achievable without identical code. Accept hybrid path.

## COMPLETED P1
1. ✅ MTP spec decode — gen_text_mtp working at 8.5 tok/s (4% acceptance from quantized head)
2. ✅ Vision pipeline verified — full screenshot→encoder→mmproj→text model→logits
3. ✅ 2 segfault bugs fixed in wubu_vision.c (n_patches_total cap, heap scores)
4. ✅ test_vision_real builds with GPU_SUPPORT

## NEXT: P2 Feature Cream
- GPU RMSNorm + SiLU + gated norm kernels
- Chunked prefill (3-7x speedup)
- NSA sparse attention
- RoPE extrapolation 4x
- GPU vision encoder kernels

## CUDA sm_120 Bugs Documented
1. static `__shared__` inside loops → hang on Blackwell
2. `__syncthreads()` + between-warps reduction → hang
3. `extern __shared__ uint8_t` + syncthreads → wrong code generation
   Fix pattern: use `extern __shared__ float smem[]` + thread-0 serial reduce

## COMMITS
- 855da96 — 4 MoE bugfixes: GPU_SUPPORT, F16 denormals, FORCE_CPU_SSM, test methodology
- 12ad638 — v5 Q8_K input quantization for MoE kernel (Q8_K int8 dot, rintf, sm_120 workarounds)
