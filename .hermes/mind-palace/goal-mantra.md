═══ GOAL PASTE (May 17 v10 — MoE expert layout bug found) ═══
PROJECT: bytropix — Custom Qwen3.6-35B-A3B inference engine
MODEL: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
┌─────────────────────────────────────────────────────────────┐
│ ARCH: qwen35moe │ 40L │ D_MODEL=2048 │ Q16/KV2 │ SSM16/32 │
│ MoE: 256 experts │ top-8 │ shared │ IQ2_XXS gate/up/down │
└─────────────────────────────────────────────────────────────☆

=== FIXED ===
1. Output projection TRANSPOSE — weight[j*D_MODEL+k]→weight[k*vocab_size+j]
   Places: infer_text.c:1374 (final), infer_text.c:1219 (GQA), wubu_model.c:429 (model API)
   Cos-sim went from -0.457 to -0.001 (anti-correlation gone, revealed deeper bug)

=== ROOT CAUSE FOUND (DA Audit) ===
2. MoE expert tensor layout WRONG — dequant_one_expert_contiguous uses contiguous
   extraction but data is INTERLEAVED by expert (innermost dim = 256 = N_EXPERTS)
   blk.0.ffn_gate_exps.weight dims = [2048, 512, 256]
   Element [i,j,e] at offset i*512*256 + j*256 + e
   Each IQ2_XXS block = ALL experts at ONE (i,j) position, not one expert

   Fix: stride-extract per expert — dequant each block, take block_vals[eid]

=== REMAINING (post-MoE-fix) ===
3. Verify full model output matches reference after MoE fix
4. Layer-by-layer comparison if still wrong
5. CPU/GPU optimizations: OpenMP on loading, VRAM cleanup SIGINT, GPU RoPE 0.25x

=== TOOLS ===
- dump_llama_logits: dumps reference logits + hidden states via llama.cpp API
- infer_text: main inference binary (NOGPU=1 for CPU, MOE=1 for MoE)
- dump_tensor_our: dumps any tensor's dims + dequantized values

=== BUILD ===
rm -f infer_text; make infer_text
