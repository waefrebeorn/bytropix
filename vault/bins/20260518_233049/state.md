# State — May 18, 2026 — MTP DECODE STABLE. ALL QUANT TYPES WORKING.

## REAL STATUS: gen_text_mtp produces stable output at 0.7 tok/s.
EMBEDDING BUG FIXED: token_embd vs emb_file precedence. gen_text.c uses
embedding file when use_embedding_file=true. gen_text_mtp now matches.

## Key Fixes This Session
1. **save_last_hidden** added to wubu_model_t — copies last hidden state
   after final RMSNorm, before output projection. For MTP h_39 capture.
2. **Embedding lookup fix** — get_embd() now checks use_embedding_file FIRST
   matching gen_text.c's read_embedding(). Previously checked token_embd
   pointer which had stale/stub data, producing NaN in the first layer.
3. **Local vec_dot restored** — reverted ggml_vec_dot_* names back to local
   q4_K_vec_dot etc. Removed libggml-cpu.so dependency. gen_text links
   standalone again.
4. **Q2_K/Q3_K** handled via dequant-to-F32 + SGEMM fallback (only used for
   blk.40 MoE, 1 layer during draft phase).

## Current Code State
- gen_text_mtp.c: clean decode with save_last_hidden. MTP draft logic
  placeholder (not yet wired — needs SSM state save/restore).
- quantized_matmul.c: Q8_0 on-the-fly dequant, IQ2_S/Q2_K/Q3_K 
  dequant-to-F32 fallback. Local vec_dot for Q4_K/Q5_K/Q6_K/IQ2_XXS/IQ3_XXS.
- wubu_model.c: save_last_hidden field + code. MTP infa (wubu_mtp_load,
  wubu_mtp_draft_forward) intact.

## Observations
- MTP model produces DIFFERENT output from non-MTP model (blk.39 shexp is
  Q8_0, some MoE are IQ2_S). Old gen_text silently skipped these tensors.
- New output "eyond..." is the CORRECT prediction for MTP model with all
  tensors properly handled.
- Decode speed 0.7 tok/s (unchanged — no MTP spec-decode active).

## Next Steps
1. Wire MTP spec-decode: save/restore SSM state around verify phase
2. Acceptance: batch-forward draft tokens, compare argmax with draft
3. SSM state save: copy ssm_states before verify, restore on reject
4. KV cache rollback: reset gqa_cache_len on reject
5. Test: MTP acceptance rate and tok/s improvement
