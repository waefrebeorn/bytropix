# Prestige Prompt — May 18, 2026 (23:30) — MTP INFRA COMPLETE

## Project: bytropix — Qwen3.6-35B-A3B-UD-IQ2_M / MTP
Cos-sim vs ref: **0.9969** — Phase 2 DONE. All 40 layers > 0.995.
gen_text coherent: "The capital of France is → the city of Paris..." (non-MTP model)
gen_text_mtp stable: 0.7 tok/s decode on MTP model (all 753 tensors handled).

## Architecture (qwen35moe → qwen35moe.cpp)
40 layers: 30 SSM (Gated DeltaNet) + 10 GQA (full attention)
Pattern: 10 × (3 SSM → 1 GQA). Hidden=2048, Vocab=248320, Expert dim=512.
SSM: 16 K-heads, 32 V-heads, d_state=128, conv_kernel=4
GQA: 16 Q-heads, 2 KV-heads, head_dim=256, IMRoPE [11,11,10,0], theta=10M
MoE: 256 experts, top-8 + 1 shared, IQ2_XXS/IQ3_XXS/IQ4_XS

## MTP Head (blk.40 + nextn)
Arch: h_39 → hnorm → concat[hnorm|enorm(embd)] → eh_proj(4096→2048,Q8_0) →
blk.40 GQA+MoE(Q5_K/Q2_K/Q3_K) → shared_head_norm → output.weight(Q4_K) → logits
Extra tensors: 20 (blk.40 + nextn.*) in MTP GGUF (753 vs 733).
MoE draft quant: Q2_K (gate/up), Q3_K (down) — lighter than main model.

## Quant Types Supported (all 16-bit aligned)
| Type | ID | Bytes/256 | Method | First Used |
|------|----|-----------|--------|------------|
| Q2_K | 10 | 84 | dequant→SGEMM | May 18 |
| Q3_K | 11 | 110 | dequant→SGEMM | May 18 |
| Q4_K | 12 | 144 | local vec_dot | May 16 |
| Q5_K | 13 | 176 | local vec_dot | May 16 |
| Q6_K | 14 | 210 | local vec_dot | May 16 |
| IQ2_XXS | 16 | 66 | local vec_dot | May 16 |
| IQ3_XXS | 18 | 98 | local vec_dot | May 16 |
| IQ2_S | 22 | 88 | dequant→SGEMM | May 18 |
| IQ4_XS | 23 | 136 | local vec_dot | May 16 |
| Q8_0 | 8 | 34/32 | on-fly dequant | May 18 |
| BF16 | 30 | 2/elem | on-fly SGEMM | May 18 |

## BUG FIXES (Historical)
1. GQA Q/gate interleave: cos-sim -0.51 → 0.9968
2. IMRoPE implemented for multi-token
3. Output proj buffer overflow fix
4. MoE OpenMP race: thread-local scratch
5. 160→5 mallocs/forward (reusable buffers)
6. Tokenizer: handle edge-case byte tokens
7. save_last_hidden corruption (none — was clean)
8. Embedding lookup: NaN fix (use_embedding_file > token_embd)

## Next: Wire MTP Spec-Decode
- SSM state save/restore for verify rollback
- Batch verify draft tokens through 40 layers
- Acceptance via argmax comparison
- Target: 1.5-2.5 tok/s
