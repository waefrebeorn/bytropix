# Prestige Prompt — May 19, 2026 (01:30) — MTP INFRA + STATE SAVE/RESTORE DONE

## Project: bytropix — Qwen3.6-35B-A3B-UD-IQ2_M / MTP
Cos-sim vs ref: **0.9969** — Phase 2 DONE. All 40 layers > 0.995.
gen_text: coherent 0.7 tok/s (non-MTP model).
gen_text_mtp: MTP=1 free-tokens mode 3.3 tok/s (IQ2_M quality loss).

## Architecture (qwen35moe → qwen35moe.cpp)
40 layers: 30 SSM (Gated DeltaNet) + 10 GQA (full attention)
Pattern: 10 × (3 SSM → 1 GQA). Hidden=2048, Vocab=248320, Expert dim=512.
SSM: 16 K-heads, 32 V-heads, d_state=128, conv_kernel=4
GQA: 16 Q-heads, 2 KV-heads, head_dim=256, IMRoPE [11,11,10,0], theta=10M
MoE: 256 experts, top-8 + 1 shared, IQ2_XXS/IQ3_XXS/IQ4_XS

## MTP Head (blk.40 + nextn) — FULLY LOADED
Arch: h_39 → hnorm(nextn_hnorm) → concat[hnorm|enorm(embd)] →
eh_proj(4096→2048,Q8_0) → blk.40 GQA+MoE(Q5_K/Q2_K/Q3_K) →
shared_head_norm → output.weight(Q4_K) → logits
Extra tensors: 20 (blk.40 + nextn.*) in MTP GGUF (753 vs 733).
MoE draft quant: Q2_K (84B/256 gate/up), Q3_K (110B/256 down).

## BUG FIXES (This Session)
9. nextn_hnorm NULL ptr: tensor found but never allocated → SIGSEGV in MTP draft. FIXED.
10. save_last_hidden: was post-final-norm (wrong for MTP). Changed to pre-final-norm. FIXED.

## Key Infrastructure
- wubu_model_checkpoint/rollback: saves/restores SSM states, conv states, KV cache lengths
- Lazy allocation: save buffers allocated on first checkpoint call
- gen_text_mtp: MTP=1 env var to enable, default non-MTP
- Non-MTP GGUF: graceful fallback (no blk.40 → skip MTP)

## Known Limitation
MTP spec-decode (verify) has 0% acceptance at IQ2_M. blk.40 MoE at Q2_K/Q3_K
quantization introduces too much noise. MTP head predictions diverge from main model.
Higher-precision MTP model needed for working spec-decode.

## Phase 7 Next: Hardware Saturation
- AVX2 vec_dot for Q4_K/Q5_K/Q6_K (256-bit SIMD)
- _mm_prefetch weight data before each layer
- OpenMP on GQA score/softmax loops
- NUMA-aware thread binding
