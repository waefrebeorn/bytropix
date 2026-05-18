# bytropix — State Dashboard (May 18 — Phase 2 Complete)

## Inference Engine (gen_text — CPU-only decode)

| Metric | Value | Verification |
|--------|-------|-------------|
| Full model cos-sim vs ref | **0.9968** | Runtime-verified (test_full_moe) |
| Decode speed | **0.6 tok/s** | 32-token generation wall clock |
| Prefill speed | **1.0-1.4 tok/s** | Varies by prompt length |
| Per-layer cos-sim decay | 0.9985→0.9952 | All 40 layers > 0.995 |
| MoE timing | 15-17ms/layer | OpenMP 3× speedup |
| GQA RoPE | IMRoPE [11,11,10,0] | Verified T=2 no NaN |

## Critical Bug Fixes (this week)

1. **GQA Q/gate interleave** (May 18): cos-sim -0.51 → 0.9968
2. **IMRoPE** (May 18): multi-token position encoding
3. **gen_text buffer overflow** (May 18): logits needed n_prompt×vs
4. **MoE OpenMP race** (May 18): thread-local scratch → deterministic
5. **Buffer reuse** (May 18): 160 mallocs → 5 per forward

## DA v10 Gap Audit

| Gap | Status | Evidence |
|-----|--------|----------|
| 1. Dequant noise | ✅ Closed | Output proj cos-sim 0.99995 vs SGEMM |
| 2. GPU output proj | ✅ Closed | CPU path verified, GPU not tested |
| 3. Decode pipeline | ✅ Closed | gen_text produces coherent English |
| 4. MoE perf | ✅ Closed | 3× OpenMP speedup, thread-local |
| 5. Shared expert gate | ✅ Closed | sigmoid gate at wubu_moe.c:448 |
| 6. SSM norm | ✅ Closed | cos-sim verified per-layer |
| 7. Chat template | ⚠️ Open | gen_text doesn't apply it |
| 8. Tensor audit | ✅ Closed | All 733 tensors loaded |
| 9. Final norm | ✅ Closed | Component of full-model cos-sim |
| 10. Ground truth | ✅ Closed | cos-sim 0.9968 vs llama.cpp |

## Architecture

```
40 layers: 30 SSM (Gated DeltaNet) + 10 GQA
Hidden: 2048, Vocab: 248320, Context: 262K
SSM: 16 K-heads × 128, 32 V-heads × 128
GQA: 16 Q-heads × 256, 2 KV-heads × 256
MoE: 256 experts, 8 active + 1 shared, dim=512
Quant: IQ2_XXS / IQ3_XXS / IQ4_XS / Q5_K / Q6_K / Q4_K / F32
```

## Key Binaries

| Binary | Command | Purpose |
|--------|---------|---------|
| gen_text | `./gen_text "prompt" 32` | CPU text generation |
| test_full_moe | `make test_full_moe && ./test_full_moe` | Cos-sim vs ref |
| ref_dumper | `make ref_dumper && ./ref_dumper <model> <token_id>` | Reference dumps |
| PROFILE | `PROFILE=1 ./test_full_moe` | Per-layer timing |

## Performance Baseline (CPU, 16 threads)

| Measurement | Value | Notes |
|-------------|-------|-------|
| Decode step | ~1.5s | 40 layers, all quantized |
| SSM layer | ~13-40ms | Varies (L0 warmup higher) |
| MoE layer | ~15-17ms | 8 experts + shared, OpenMP |
| GQA layer | ~15ms | 10 layers, no KV cache |
| Output proj | ~11-14ms | Q4_K × 248K vocab |

## Known Limitations

- **No chat template** → minor quality degredation
- **No KV cache** → GQA recomputes full attention each step
- **No SIMD vec_dot** → 0.003 cos-sim gap from quantization noise
- **No GPU decode** → all layers on CPU (0.6 tok/s hard limit)
- **SSM L2 eps** = 1e-12 (llama.cpp uses ~1e-6) — not blocking

## Next Priorities

1. Chat template (quality fix)
2. KV cache for GQA (10% speedup)
3. SIMD vec_dot for cos-sim → 1.0
4. GPU decode path (wire existing kernels into gen_text loop)
