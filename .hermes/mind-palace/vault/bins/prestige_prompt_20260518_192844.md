# Prestige Prompt — May 18, 2026 — PHASE 2 COMPLETE

## Project: bytropix — Qwen3.6-35B-A3B-UD-IQ2_M
Cos-sim vs ref: **0.9969** — Phase 2 DONE. All 40 layers > 0.995.
gen_text coherent: "The capital of France is → the city of Paris..."
Decode: 0.7 tok/s (2.3× faster). DA gaps: **ALL 10 CLOSED**.

## Architecture (qwen35moe → qwen35moe.cpp)
40 layers: 30 SSM (Gated DeltaNet) + 10 GQA (full attention)
- Pattern: 10 × (3 SSM → 1 GQA)
- Hidden: 2048, Vocab: 248320, Expert dim: 512, Shared dim: 512
- SSM: 16 K-heads, 32 V-heads, d_state=128, conv_kernel=4
- GQA: 16 Q-heads, 2 KV-heads, head_dim=256
- RoPE: IMRoPE, sections [11,11,10,0], freq_base=10M ✓
- MoE: 256 experts, top-8 + 1 shared, IQ2_XXS/IQ3_XXS/IQ4_XS

## What's Wired
- SSM: quantized_matmul (Q5_K/Q6_K) ✓
- GQA: quantized_matmul (Q5_K) ✓ w/ interleave fix + IMRoPE
- MoE: quantized_matmul (IQ2_XXS/IQ3_XXS/IQ4_XS + Q5_K/Q6_K shared) ✓
- sigmoid gate for shared expert ✓ (DA Gap 5 = already fixed)
- Output: Q4_K quantized matmul ✓ (cos-sim 0.99995 vs SGEMM)
- gen_text: full prefill+decode pipeline ✓

## BUG FIXES
1. GQA Q/gate interleave: cos-sim -0.51 → 0.9968
2. IMRoPE implemented for multi-token
3. Output proj buffer overflow fix
4. MoE OpenMP race: thread-local scratch
5. 160→5 mallocs/forward (reusable buffers)
6. Tokenizer: handle edge-case byte tokens

## Remaining DA Gaps
- Gap 7 (chat template): **CLOSED** — CHAT=1 env var ✓
- Gap 10 (ground truth): CLOSED — cos-sim 0.9969

## Next: Phase 5 — Island Boy + MTP Speculative Decode
Memory bandwidth is bottleneck (10.7GB/step, DDR5 ~50GB/s).
Batch tokens per layer (B=4) to amortize weight load.
MTP head (blk.40 + nextn) for 3-4 token draft, batch verify.
Accept 5-token startup lag for cache warmup.
Target: 2-3 tok/s.

## Performance (CPU, 16 threads)
- Decode: 0.6 tok/s
- Prefill: 1.4 tok/s
- MoE: 15ms/layer, SSM: 13ms, GQA: 15ms
