# Overnight Map — May 18, 2026 (23:30) — MTP INFRA COMPLETE

## Session Summary
Implemented ALL missing quant types (Q2_K, Q3_K, Q8_0, IQ2_S, BF16) in
quantized_matmul.c via dequant-to-F32 fallback. Added save_last_hidden field
for h_39 capture. Fixed NaN bug: embedding lookup precedence (use_embedding_file
over token_embd). MTP model loads fully. gen_text_mtp stable at 0.7 tok/s.

## Done
1. save_last_hidden on wubu_model_t — captures last hidden state after RMSNorm,
   before output projection. Set to any [D_MODEL] buffer before forward call.
2. Q2_K/Q3_K/Q8_0/IQ2_S/BF16 — all added to quantized_matmul dispatch.
   Q8_0: on-the-fly F16→F32 dequant + SGEMM per column.
   Q2_K/Q3_K/IQ2_S: dequant entire weight to F32 + SGEMM.
   BF16: on-the-fly BF16→F32 shift + SGEMM.
3. Local vec_dot names RESTORED (reverted libggml-cpu.so dependency).
4. gen_text_mtp.c: save_last_hidden for prefill + decode. MTP draft logic
   placeholder (needs SSM state save/restore).
5. Embedding bug: get_embd() checked token_embd before use_embedding_file.
   token_embd had stale stub data → NaN on first decode step. FIXED.

## Key Files Changed
- include/gguf_reader.h: GGML_TYPE_BF16 = 30
- include/wubu_model.h: save_last_hidden field, mtp_head_t restored
- src/gguf_reader.c: BF16 dequant + raw_size
- src/quantized_matmul.c: Q8_0 on-fly, Q2_K/Q3_K/IQ2_S/BF16 fallback
- src/wubu_model.c: save_last_hidden capture, MTP load/draft/free intact
- tools/gen_text_mtp.c: stable decode tool
- Makefile: gen_text_mtp target

## Next Session
1. Allocate SSM state save buffers in wubu_model_t
2. In gen_text_mtp: save SSM state + KV cache len before verify forward
3. Call wubu_mtp_draft_forward for draft tokens
4. Batch verify through 40 layers
5. Acceptance: compare verify argmax vs draft, accept longest prefix
6. Rollback SSM/KV on partial reject
7. Benchmark tok/s with MTP enabled

## Reference
- vault/unsloth-quantization-format.md — UD format docs
- ref_dumper — links libllama.so for reference data
- gen_text — standard decode (non-MTP model)
- gen_text_mtp — MTP model decode with save_last_hidden
