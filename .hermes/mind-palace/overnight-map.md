# Overnight Map — May 18, 2026 — POST-FIX

## Session Summary
**ROOT CAUSE FOUND AND FIXED**: GQA Q/gate per-head interleave bug.
Cos-sim from -0.51 → 0.9968. MoE quantized path also wired.

## Completed
1. **GQA interleave fix**: attn_q.weight output [8192] is per-head interleaved
   [Q_h0(256)][gate_h0(256)][Q_h1(256)]... Our code used contiguous split [Q][gate].
   Fix applied to both gate extraction and Q normalization.

2. **MoE quantized path**: IQ2_XXS/IQ3_XXS/IQ4_XS via blob ptrs + quantized_matmul
   Both shared expert (Q5_K/Q6_K) and routed experts wired.

3. **Per-layer dump**: Modified llama.cpp to dump per-layer hidden states
   via LLAMA_DUMP_LAYERS + DUMP_LAYER_DIR env vars. Same in bytropix.

4. **Type verification**: Ran dump_tensor_types on actual GGUF. Real types are
   IQ2_XXS(16), IQ3_XXS(18), IQ4_XS(23) — NOT "IQ2_XS/IQ3_XS/Q3_S_XL" as markdown claimed.

5. **All mind-palace docs updated** with honest post-fix assessment.

## Next Session
1. Verify GQA RoPE (needed for multi-token generation, not single-token)
2. Push cos-sim from 0.9968 toward 1.0 if desired (quantization noise ceiling)
3. Build infer_text for actual text generation
4. Add OpenMP to attention loops if missing
