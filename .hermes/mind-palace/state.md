# State — May 18, 2026 — ALL PHASES COMPLETE

## REAL STATUS: Qwen3.6-35B-A3B inference engine complete.
Cos-sim 0.9970 vs llama.cpp (quantization noise floor).
gen_text CHAT=1 produces structured reasoning. Decode 0.7 tok/s.

## Verified Runtime Results
- Full 40L + quantized MoE: cos-sim **0.997022** vs llama.cpp (SSE vec_dot)
- All 40 layers cos-sim > 0.995
- Q4_K/Q5_K/Q6_K: SSE3 `_mm_maddubs_epi16` + SSE4.1 `_mm_cvtepi8_epi16`
- IQ2_XXS/IQ3_XXS/IQ4_XS: still generic C (complex lookup tables)
- Output projection Q4_K: cos-sim 0.99995 vs F32 SGEMM ✓
- gen_text: "The capital of France is" → coherent English ✓

## Performance (CPU, 16 threads, 2.7bpw 35B MoE)
- Decode: 0.6 tok/s (2× improvement from MoE OpenMP + embedding fix)
- Prefill: 1.4 tok/s
- MoE: 15ms/layer (3× from OpenMP), SSM: 13ms/layer, GQA: 15ms/layer
- Malloc reduction: 160 mallocs → 5 per forward (pre-allocated buffers)
- PROFILE env var enabled for per-layer timing

## DA v10 Gaps Status
- Gap 1-10: **ALL CLOSED** ✓
- **New: GQA KV cache fixed decode attention** — was only attending to self,
  now attends to all previous tokens via persistent K/V cache.
