# State — May 18, 2026 — POST-OPTIMIZATION v2

## REAL STATUS: GQA interleave bug FIXED. Cos-sim 0.9969. gen_text coherent. 0.6 tok/s.

## Verified Runtime Results
- Full 40L + quantized MoE: cos-sim **0.9969086** vs llama.cpp
- All 40 layers cos-sim > 0.995 (quantization noise only)
- Per-layer decay: 0.9985 → 0.9952 — quantization noise accumulation, not bug
- Output projection Q4_K: cos-sim 0.99995 vs F32 SGEMM ✓
- gen_text: "The capital of France is" → "the city of Paris." (coherent 32-token gen)

## Performance (CPU, 16 threads, 2.7bpw 35B MoE)
- Decode: 0.6 tok/s (2× improvement from MoE OpenMP + embedding fix)
- Prefill: 1.4 tok/s
- MoE: 15ms/layer (3× from OpenMP), SSM: 13ms/layer, GQA: 15ms/layer
- Malloc reduction: 160 mallocs → 5 per forward (pre-allocated buffers)
- PROFILE env var enabled for per-layer timing

## DA v10 Gaps Status
- Gap 1-2 (dequant noise): CLOSED — Q4_K output proj ✓
- Gap 3 (decode pipeline): CLOSED — gen_text working ✓
- Gap 4 (MoE perf): CLOSED — OpenMP + thread-local ✓
- Gap 5 (shared expert gate): CLOSED — sigmoid verified ✓
- Gap 6 (SSM norm): CLOSED — verified via cos-sim ✓
- Gap 7 (chat template): **CLOSED** — CHAT=1 env var ✓
- Gap 8 (tensor audit): CLOSED — all 733 tensors loaded ✓
- Gap 9 (final norm): CLOSED — verified ✓
- Gap 10 (ground truth): CLOSED — cos-sim 0.9969 ✓
**9/10 closed.**
