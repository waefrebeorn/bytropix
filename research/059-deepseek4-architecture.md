# research/059 — DeepSeek-V4-Flash-0731 Config-I: the architecture, from the file

> 2026-08-04. The REAL headers of all three `DeepSeek-V4-Flash-0731-ConfigI` GGUF
> parts (thetom-ai, turboquant branch) parsed by our own loader (test_gguf_load)
> after the hive fixes. THE FILE IS THE TRUTH: the artifact is **43 layers,
> 284.3B params, 1328 tensors** across the 3 split files — matching the
> README/card (284B). Each split carries a DISJOINT tensor set: part 1 = blk.0-18
> (566 tensors), part 2 = blk.18-36 (592), part 3 = blk.37-42 + head (170).
> (A first pass over part 1 alone looked like "19 blocks / 121.9B" — a
> split artifact, corrected by merging all three headers.)

## Verified facts (from the GGUF headers)

- **Total params: 284.3B** (sum of all 1328 tensor shapes) — README's 284B ✓.
- **43 blocks** (blk.0..blk.42). The tensor set per block varies by split —
  e.g. blk.18's attn lives in part 1 while its ffn gate/up live in part 2.
- hidden = 4096, vocab = 129280 (token_embd + output both (4096,129280)).
- **Type histogram (7 types, zero unknown across all parts):**
  F32 272 · TQ3_1S 45 · Q2_0 47 · BF16 30 · I32 26 · Q8_0 8 · Q6_K 14

## Per-layer layout (blk.0 as the template)

| tensor | dims | type | role |
|---|---|---|---|
| attn_norm | (4096,) | F32 | RMS pre-norm |
| attn_q_a | (4096,1024) | TQ3_1S | MLA q latent down |
| attn_q_a_norm | (1024,) | F32 | |
| attn_q_b | (1024,32768) | TQ3_1S | q up → 256 heads × 128 |
| attn_kv | (4096,512) | TQ3_1S | KV latent (kv_lora 512) |
| attn_kv_a_norm | (512,) | F32 | |
| attn_output_a | (4096,8192) | TQ3_1S | o latent up |
| attn_output_b | (8192,4096) | TQ3_1S | o down |
| attn_sinks | (64,) | F32 | sink tokens |
| attn_compressor_ape/gate/kv/norm | — | TQ3_1S/F32 | KV compressor (17 layers) |
| ffn_norm | (4096,) | F32 | |
| ffn_gate_inp | (4096,256) | BF16 | router logits |
| ffn_gate_tid2eid | (6,129280) | I32 | hash-router TID→EID (blk 0..2) |
| ffn_gate_exps | (4096,2048,256) | Q2_0 | 2.15B |
| ffn_up_exps | (4096,2048,256) | Q2_0 | 2.15B |
| ffn_down_exps | (2048,4096,256) | TQ3_1S | 2.15B |
| ffn_gate_shexp/up_shexp/down_shexp | (4096,2048) | TQ3_1S | shared expert |
| exp_probs_b.bias | — | F32 | expert probability bias (16 layers) |
| hc_attn_base/fn/scale | (24,)/(16384,24)/(3,) | F32 | mHC hyper-connection |
| hc_ffn_base/fn/scale | same | F32 | mHC |

## Forward map onto in-tree modules (all building blocks EXIST)

1. embd Q8_0 → `gguf_read_tensor_f32` (Q8_0 dequant done)
2. 19 × MLA — `src/wubu_mla.c` (q_a/q_b/kv latent, head_dim 128)
3. MoE 256×top-6 + shared — `src/wubu_moe.c` / `wubu_moe2.c`
4. hash routing (tid2eid) — `src/wubu_hashrouter.c` (I32 table direct)
5. DSA indexer — `src/wubu_dsa.c` (coarse-to-fine)
6. mHC hyper-connections — `src/wubu_mhc_mh.c` (24-dim bodies)
7. KV compressor + sinks — new small module
8. head Q6_K — dequant done (hive-fixed f16 subnormal)

## Remaining PACE work

- [x] type remap (47/42→Q2_0, TQ3_1S=45, TQ4_1S=46) + exact dequants (unit-tested)
- [x] KV value-type table (hive 2nd catch) + I8..F64 support
- [x] load gate on the REAL header: 566 tensors, 0 unknown, 0 span mismatches
- [ ] full 3-split load (download running; split data redirection = next loader step)
- [ ] wubu_deepseek4.c forward (wiring the above — REAL ARCH WORK, next phase)
- [ ] sampling: temp 1.0 / top_p 0.95, FP16 KV, DRY/repeat-penalty for math
- [ ] generation gate + tokens/s
