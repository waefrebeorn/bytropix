# Overnight Map — May 19, 2026 (01:30) — MTP INFRA + STATE SAVE/RESTORE DONE

## Session Summary
Implemented SSM state checkpoint/rollback for speculative decode. Fixed nextn_hnorm
NULL pointer bug (load bug from previous session). Implemented full MTP spec-decode
with per-token verify and rollback. Discovered 100% rejection rate at IQ2_M — MTP
head (Q2_K/Q3_K MoE) too noisy for verify. Pivoted to MTP "free-tokens" mode
(no verify, opt-in via MTP=1 env var). Default gen_text_mtp runs as non-MTP.

## Done
1. wubu_model.h: added ssm_states_saved, conv_states_saved, cache_len saved fields
2. wubu_model_checkpoint(): lazy-alloc save buffers, copies SSM/conv states + cache lens
3. wubu_model_rollback(): restores SSM/conv states + cache lens from save buffers
4. Free save buffers in wubu_model_free()
5. nextn_hnorm load bug fix: was found but never malloc'd → SIGSEGV
6. save_last_hidden: changed to capture pre-final-norm (correct for MTP)
7. gen_text_mtp: full spec-decode loop with per-token verify/rollback
8. gen_text_mtp: MTP free-tokens mode (no verify, emit MTP outputs)
9. gen_text_mtp: MTP=1 opt-in env var, default non-MTP
10. All mind-palace files updated with findings

## Key Files Changed
- include/wubu_model.h: save/restore fields + checkpoint/rollback decl
- src/wubu_model.c: checkpoint/rollback impl, nextn_hnorm load fix, save_last_hidden pre-norm
- tools/gen_text_mtp.c: full rewrite with MTP spec-decode + free-tokens mode

## Next Session
1. Phase 7: Hardware saturation (AVX2 vec_dot, prefetch, OpenMP on GQA)
2. Higher-precision MTP model for working verify
3. MTP with non-MTP GGUF already gracefully handled

## Key Insight
MTP spec-decode at IQ2_M is ineffective. 100% rejection rate. The MTP head's
blk.40 MoE at Q2_K/Q3_K is too aggressively quantized to agree with the
IQ2_M-quantized main model. Speedup from MTP free-tokens mode comes at quality
cost. For production MTP, need Q4_K_M or better for blk.40.

## Reference
- vault/unsloth-quantization-format.md — UD format docs
- gen_text — standard decode (0.7 tok/s)
- gen_text_mtp — MTP model decode (MTP=1 for free-tokens mode)
