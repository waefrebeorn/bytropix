# State — May 18, 2026 (23:30) — ALL QUANT TYPES SUPPORTED. MTP INFRA COMPLETE.

## REAL STATUS: gen_text_mtp stable at 0.7 tok/s. MTP spec-decode logic placeholder.
Phiases: 0-4 DONE (core inference, DA gaps, SIMD vec_dot, KV cache). Phase 5+6: MTP
model loads, blk.40 + nextn head accessible via wubu_mtp_load/wubu_mtp_draft_forward.
save_last_hidden captures h_39. Embedding lookup fixed (NaN root cause).

## Quant Type Support (quantized_matmul.c dispatch)
| Type | ID | Strategy | Used By |
|------|----|----------|---------|
| F32 | 0 | Direct SGEMM | Norms, routers |
| Q4_K | 12 | Local vec_dot (q4_K_vec_dot) | output.weight |
| Q5_K | 13 | Local vec_dot (q5_K_vec_dot) | attn q/k/v, shared gate/up, token_embd |
| Q6_K | 14 | Local vec_dot (q6_K_vec_dot) | SSM output proj, shared down |
| IQ2_XXS | 16 | Local vec_dot (iq2_xxs_vec_dot) | MoE gate/up experts |
| IQ3_XXS | 18 | Local vec_dot (iq3_xxs_vec_dot) | MoE down experts |
| IQ4_XS | 23 | Local vec_dot (iq4_xs_vec_dot) | MoE down (L34,38,39) |
| Q8_0 | 8 | On-the-fly dequant+SGEMM | blk.39 shexp, nextn.eh_proj |
| Q2_K | 10 | Dequant-to-F32+SGEMM | blk.40 MoE gate/up (MTP head) |
| Q3_K | 11 | Dequant-to-F32+SGEMM | blk.40 MoE down (MTP head) |
| IQ2_S | 22 | Dequant-to-F32+SGEMM | MTP model some layers |
| BF16 | 30 | On-the-fly SGEMM | blk.40 router tensors |

## Key Code Changes (this session)
- include/gguf_reader.h: GGML_TYPE_BF16 = 30 in enum
- include/wubu_model.h: save_last_hidden field, mtp_head_t restored
- src/gguf_reader.c: BF16 dequant + raw_size
- src/quantized_matmul.c: Q8_0 on-fly dequant, Q2_K/Q3_K/IQ2_S/SGM fallback
- src/wubu_model.c: save_last_hidden capture. MTP load/draft/free intact
- tools/gen_text_mtp.c: save_last_hidden for h_39. Embedding lookup bugfix.
- Makefile: gen_text_mtp target

## MTP Model Architecture
h_39 → rms_norm(nextn.hnorm) → concat[hnorm(h_39) | enorm(embd)] →
eh_proj(4096→2048, Q8_0) → blk.40 GQA+MoE (Q5_K/Q2_K/Q3_K)→
rms_norm(nextn.shared_head_norm) → output.weight(Q4_K) → logits

## Performance
- Prefill: 2.1 s/5tok (2.4 tok/s)
- Decode: 0.7 tok/s (CPU, 16 threads, 35B MoE)
- MoE: 15ms/layer, SSM: 13ms/layer, GQA: 15ms/layer
- Memory bandwidth bottleneck: 10.7GB/step, DDR5 ~50GB/s

## Next: MTP Spec-Decode Implementation
1. SSM state save/restore arrays (allocate buffers, copy before verify)
2. KV cache rollback (save/restore gqa_cache_len)
3. Batch verify: forward draft tokens (B=3-4) through 40 layers
4. Acceptance: compare verify argmax vs draft tokens per position
5. Accept longest prefix, rollback rest
6. Target: 1.5-2.5 tok/s via MTP (~80% acceptance rate predicted)
