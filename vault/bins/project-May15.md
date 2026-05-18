# bytropix — Project Overview (May 18 — Phase 2 Complete)

## Mission
Build Qwen3.6-35B-A3B inference from scratch in C + CUDA.
**Phase 2 complete:** 0.6 tok/s decode, cos-sim 0.9968 vs llama.cpp.

## Components

| Component | Implementation | Status |
|-----------|---------------|--------|
| GGUF reader | `gguf_reader.c` — 7 dequant types | ✅ In use |
| SSM forward | `wubu_ssm.c` — Gated DeltaNet | ✅ Verified |
| GQA forward | `wubu_ssm.c` — IMRoPE + attention | ✅ Verified |
| MoE forward | `wubu_moe.c` — router + experts | ✅ Verified |
| Output proj | `quantized_matmul.c` — Q4_K matmul | ✅ cos-sim 0.99995 |
| gen_text | `tools/gen_text.c` — full pipeline | ✅ Working |
| ref_dumper | `tools/ref_dumper.cpp` — libllama.so | ✅ Ground truth |

## Key Achievements (May 18)

- **GQA Q/gate interleave bug FIXED**: cos-sim -0.51 → 0.9968
- **IMRoPE implemented**: sections [11,11,10,0], theta=10M
- **MoE OpenMP**: 3× speedup (44ms→15ms per layer)
- **Buffer reuse**: 160 mallocs → 5 per forward pass
- **gen_text**: coherent English generation at 0.6 tok/s
- **DA v10 gaps**: 8/10 closed. Chat template remains.

## Performance

| Metric | Value | Target |
|--------|-------|--------|
| Decode | 0.6 tok/s | >1 tok/s |
| Prefill | 1.0-1.4 tok/s | >5 tok/s |
| Cos-sim | 0.9968 | 1.0 (requires SIMD vec_dot) |
| Per-layer | all > 0.995 | all > 0.999 |

## Remaining

| Issue | Priority | Notes |
|-------|----------|-------|
| Chat template | P0 | Quality fix for gen_text |
| Multi-token verify | P0 | T>1 cos-sim unknown |
| KV cache | P1 | ~10% decode speedup |
| SIMD vec_dot | P1 | cos-sim → 1.0 |
| GPU decode | P2 | 5-10× speedup expected |
| WuBu geometry | P3 | Research code only |
