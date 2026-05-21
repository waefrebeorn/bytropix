# Overnight Map — Phase 29a: IQ1_M + Q4_K GPU Kernels

**Active repo:** /home/wubu/bytropix/  
**Current commit:** c0254c0 (not pushed)  
**Default model:** /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (11.5GB)  
**IQ1_M model:** /models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf (7.7GB, 1.90 BPW, quality degraded)

## Session Summary (May 21, 2026)

### What Was Done
1. **GPU IQ1_M quant matmul kernel** — single-token + batched variants with iq1s_grid lookup
   - Verified exact match vs CPU (max diff 3.3e-7)
   - Grid table uploaded via `wubu_cuda_quant_matmul_set_iq1s_grid` at GPU init
2. **GPU Q4_K quant matmul kernel** — single-token + batched (simpler than Q5_K, no qh field)
3. **CPU `quantized_matmul_from_q8` IQ1_M fallback** — dequant+SGEMM for types without vec_dot
4. **`MODEL` env var** — `gen_text.c` now reads `MODEL` env var to override model path

### GPU Kernel Inventory
Now supporting 4 quant types on GPU: Q5_K, Q6_K, Q4_K, IQ1_M
All have single-token and batched (C=N prefill) variants.

### Remaining GPU Blockers
1. **GPU MoE divergence** (0.9888 cos-sim per layer, DA v13) — fundamental code-path diff
2. **Q2_K, IQ2_XXS GPU kernels** — needed for token_embd, ffn_down, attn_output weights
3. **Full GPU inference** — requires all weight types and solving H2D/D2H overhead

### Next Session Options
1. Debug IQ1_M model quality — investigate why 1.90 BPW gives garbage with multi-token prompts
2. Add GPU Q2_K kernel for token_embd.weight (D_MODEL=2048, vocab=248320) and ffn_down weights
3. Fix the GPU MoE divergence root cause from DA v13
4. Re-quantize IQ1_M with more imatrix chunks for better quality
