# State — Phase 28 DA: GPU_SUPPORT Live but Unverified, F32 Dead Weight

**bytropix: inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE)**
**GPU: GQA accel active. SSM GPU path compiles and runs — correctness UNVERIFIED.**

## DA Audit Findings (May 21, 3-pass)

### ⚠️ Finding 1: F32 dequant weights on GPU waste ~2.2 GB VRAM
`wubu_model_gpu_init()` uploads BOTH quantized (Q5_K/Q6_K) AND F32 dequantized SSM weights for each layer. The 4532 MB figure is DOUBLE-COUNTED: ~2266 MB quant + ~2266 MB F32 dequant. The F32 weights are referenced only by `wubu_model_gpu_ssm_project()` which uses the OLD broken column-major quant_matmul kernel (dead code). The new `wubu_model_gpu_ssm_forward_full()` uses only the quantized weights via row_major kernel. **The F32 dequant weights are never used and never freed — memory leak.**

### ⚠️ Finding 2: Prefill N>1 fallback uses broken quant matmul
`wubu_model_gpu_ssm_project()` (line 864 in wubu_model_gpu.cu) calls `wubu_cuda_quant_matmul()` — the OLD column-major kernel with wrong stride. This function IS called from the N>1 prefill fallback path (wubu_model.c line 524). The old kernel reads garbage data. **SSM prefill at N>1 produces garbage when forward_full fails.**

### ⚠️ Finding 3: Quantized SSM weights never freed
`d_attn_qkv_q[40]`, `d_attn_gate_q[40]`, `d_ssm_out_q[40]`, `d_qkv_f32[40]`, `d_gate_f32[40]`, `d_out_f32[40]` have ZERO free calls in `wubu_model_gpu_free()`. Memory leak: ~5.5 GB of GPU allocations never freed.

### ⚠️ Finding 4: Phase 26 fused kernels verified but path untested end-to-end
The `ssm_beta_alpha_fused_decode` and `ssm_conv_silu_split_decode` kernels are called from `wubu_model_gpu_ssm_forward_full()` (lines 1068, 1085). They pass f32* device pointers. But the ENTIRE forward_full output has NEVER been compared against the CPU path. Verifying individual kernels ≠ verifying the full pipeline.

### ⚠️ Finding 5: README.md claims are stale
- "Phase: 25" — actually Phase 28
- "GPU decode 8.5 tok/s" — measured BEFORE GPU_SUPPORT was live (SSM on CPU). With SSM GPU path active, speed unknown
- "SSM beta/alpha fused kernel ✅ verified cos-sim 1.0" — verified in isolation, never in full inference
- SSM weight VRAM budget says "692 MB" but actual upload is 4532 MB (includes F32 dead weight)

## What Works (✅ actually verified at runtime)
- GPU init completes (GQA 1040 MB, SSM quant 2266 MB, KV cache 1440 MB, MoE ~460 MB) = ~5.2 GB total
- GQA attention on GPU
- Output projection on GPU (Q4_K kernel)
- wubu_gpu_set_ssm_hybrid() helper for hybrid GPU recurrence mode
- gen_text_gpu builds, runs, exits 0

## What's Broken or Unverified (❓)
- SSM GPU path output NEVER compared vs CPU path (no cos-sim done)
- F32 dequant SSM weights wasting ~2.2 GB VRAM, never freed
- Prefill N>1 fallback uses broken column-major quant_matmul
- gen_text.c hardcoded 1-token prompt (pre-existing, blocks proper testing)
- 256k context cos-sim vs llama.cpp: never done

## Verification Priority
1. **Add `-DNO_F32_UPLOAD` or remove F32 dequant upload** to save 2.2 GB VRAM
2. **Fix wubu_model_gpu_free** to free d_attn_qkv_q, d_attn_gate_q, d_ssm_out_q, and F32 variants
3. **Build proper test harness** — modify gen_text.c to accept prompt from stdin or args
4. **Compare GPU SSM vs CPU SSM** — cos-sim at single layer, then full 30 layers
5. **Fix prefill N>1 path** — use row_major kernel in wubu_model_gpu_ssm_project() too
