# Plan — May 19, 2026 (Triple DA v6 — Precision, Not Algorithm)

## TRIPLE DA ROOT CAUSE
The SSM recurrence math is IDENTICAL between bytropix and llama.cpp.
`ggml_gated_delta_net` kernel: `scale=1/sqrt(128)`, `exp(gate)` decay, same `h←h*exp(gate)+k·(v-h·k)·beta`, same `out=h·q*scale`.
Divergence is from **quantized matmul dequant precision**, not algorithm.

## Phase 9: Quantized Matmul Precision Parity [CURRENT P0]
| Sub-phase | Status | Detail |
|-----------|--------|--------|
| 9.1 Per-layer cos-sim infrastructure | ✅ DONE | DUMP_LAYER_DIR in both, ggml_set_output fix |
| 9.2 SSM recurrence verified identical | ✅ DONE | By reading ggml_gated_delta_net kernel source |
| 9.3 Measure token-level divergence | 🔜 NEXT | Run 5-token sequence, compare per-layer drift |
| 9.4 Port ggml dequant kernels to bytropix | 🔜 P0 | Q5_K, Q6_K, IQ2_XXS, IQ3_XXS for bit-exact values |

## Phase 10: Hardware Saturation [P0]
| Sub-phase | Status | Detail |
|-----------|--------|--------|
| 10.1 AVX2 IQ3_XXS vec_dot fix | ❌ BROKEN | _mm_hadd_epi16 incorrect, reverted to generic |
| 10.2 OpenMP output proj split | 🔜 P1 | Parallelize Q4_K [2048]×[2048,248320] across 16 threads |
| 10.3 Expert prefetch integration | 🔜 P1 | API ready, needs _mm_prefetch wiring |

## Phase 11: 256K Context Readiness [P1]
- SSM attn AVX2 optimization (0.8ms/layer, 24ms total — low priority)
- GQA KV cache scaling for 256K
- Chunked prefill
