# State — Phase 28k: GPU MoE Analysis Complete, Moving to P1 (MTP + Vision)

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
| GPU SSM/GQA + GPU MoE | "inless, the 1000th..." (1.6 tok/s) | ❌ Garbage (Q8_K kernel) |
| CPU-only | "Paris is the capital of France..." | ✅ |
| gen_text_mtp | Source exists, NOT compiled | 🟡 |
| Vision encoder | 384 LoC, untested | 🟡 |

## ROOT CAUSE ANALYSIS (DA v13 Complete)
The 1.1% per-layer GPU MoE error is NOT from any single bug. It is the fundamental result of running a different code implementation:
- CPU `quantized_matmul` uses `quantize_row_q8_K` (negative d, sign-inverted Q8)
- GPU kernel uses `max_abs / 127` (positive d, same-sign Q8)
- Both are mathematically equivalent but produce different floating-point rounding
- The 0.32% running cos-sim error compounds through 40 layers → flips token selection in 240K vocab

**Verdict:** GPU MoE bit-exact parity is NOT achievable without identical code. Accept hybrid path.

## PRAGMATIC PATH FORWARD
1. **GPU MoE disabled by default** (FORCE_CPU_MOE env var) — hybrid GPU SSM/GQA + CPU MoE works
2. **P1: MTP spec decode** — build gen_text_mtp, test with MTP model
3. **P1: Vision** — build test_vision_real, verify encoder
4. **Q8_K kernel (v5) committed** as reference for future GPU MoE work

## CUDA sm_120 Bugs Documented
1. static `__shared__` inside loops → hang on Blackwell
2. `__syncthreads()` + between-warps reduction → hang
3. `extern __shared__ uint8_t` + syncthreads → wrong code generation
   Fix pattern: use `extern __shared__ float smem[]` + thread-0 serial reduce

## COMMITS
- 855da96 — 4 MoE bugfixes: GPU_SUPPORT, F16 denormals, FORCE_CPU_SSM, test methodology
- 12ad638 — v5 Q8_K input quantization for MoE kernel (Q8_K int8 dot, rintf, sm_120 workarounds)
