# Prestige Prompt — May 19, 2026 (02:45) — TRIPLE DA AUDIT

## Project: bytropix — Qwen3.6-35B-A3B-UD-IQ2_M
Phase 7 complete. Triple DA audit: 7/7 key claims ✅, 2 stale ❓, 1 finding.

## Benchmarks (DA verified)
| Metric | Value | Status |
|--------|-------|--------|
| Decode | 2.1 tok/s | ✅ Verified 15.08s/32tok |
| Prefill | 7.7 tok/s | ✅ Verified 2.72s/21tok |
| Output proj decode | 6.4ms | ✅ Verified PROFILE=1 |
| MoE decode/layer | 10ms | ✅ Verified PROFILE=1 |
| llama dep free | yes | ✅ ldd+nm verified |
| cos-sim | 0.9969 | ❓ Stale (Phase 2) |

## Phase 7 Opts
- GQA attn: stack buf + AVX2 FMA (Q·K dot 4× unrolled, V sum 8-elem)
- AVX2 vec_dot: Q4_K/Q5_K/Q6_K (256-bit, 2× SSE)
- Prefetch next column in quantized_matmul
- Llama deps killed

## DA Finding: MoE Softmax Gating
Router uses softmax over all 256 experts then top-8 → renormalize. Works but DeepSeek recommends normalized sigmoid for stability + efficiency.

## Cold Gaps
| Prio | Gap | Impact |
|------|-----|--------|
| P0 | AVX2 IQ2_XXS/IQ3_XXS vec_dot | MoE = 10ms/layer = primary bottleneck |
| P1 | Normalized sigmoid gating | Efficiency + stability |
| P1 | NV64 ring buffer impl | Cache miss latency hiding |
| P2 | cos-sim re-verify, MTP higher-precision | Stale claims, working spec-decode |

## Vault Papers Read
- Unsloth UD quant formula: per-tensor bpw breakdown
- Qwen3: 256-expert MoE + thinking mode validated
- DeepSeek-V3: MTP self-spec decode, sigmoid gating
- Synthesis: P0-P3 priority map