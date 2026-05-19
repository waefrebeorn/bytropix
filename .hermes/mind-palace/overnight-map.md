# Overnight Map — May 19, 2026 Late PM (Phase 15 Complete)

## BREAKTHROUGH: Integrated GPU GQA Attention

Phase 15 wired GPU-accelerated GQA attention into the standard model forward pass. No more standalone infer_text_gpu — `GPU=1 ./gen_text_gpu` now uses GPU for GQA transparently.

### Architecture

```
wubu_model_t.gpu_ctx (opaque void* → gpu_ctx_t in wubu_model_gpu.cu)
├── cublasHandle_t + cudaStream_t (TF32 tensor cores)
├── gpu_gqa_layer_t[40] — F32 dequant weights (1.04 GB)
│   ├── d_attn_q [2048, 8192] — fused Q+gate
│   ├── d_attn_k [2048, 512]
│   ├── d_attn_v [2048, 512]
│   ├── d_attn_out_w [4096, 2048]
│   └── d_q_norm_w, d_k_norm_w [256]
├── float* d_k_cache[40], d_v_cache[40] — persistent GPU KV caches
├── float* d_sincos [max_ctx, 64] — MRoPE table
└── Scratch: d_x, d_scr, d_ktmp, d_vtmp, d_qtmp, d_gout, d_score_scr
```

### Forward Flow (per GQA layer, single token)
```
normed [2048] → upload to GPU
Q = x @ d_attn_q^T  [1,8192] fused Q+gate
K = x @ d_attn_k^T  [1,512]
V = x @ d_attn_v^T  [1,512]

Copy Q → d_qtmp (contiguous, stride 4096)
Copy gate → d_scr (contiguous, overwrite Q area)

RMSNorm Q (d_qtmp, 16 heads × 256 dim)
RMSNorm K (d_ktmp, 2 heads × 256 dim)
RoPE Q + K (MRoPE sections [11,11,10,0])

Append K,V → persistent cache at cache_len[L]

chunked_attn:
  scores = Q @ K_cache^T  [1, T_cache]
  softmax(scores) → attn_w
  out = attn_w @ V_cache  [1, 512]
  gate: out *= sigmoid(gate)
  output_proj: out @ d_attn_out_w^T  [1, 2048]

Download result → host attn_out buffer
```

### Bugs Fixed in Phase 15
1. **RMSNorm Q stride**: Q in fused buffer has stride 8192 (q_dim_x2), but RMSNorm expects stride 4096 (q_dim). Fixed by copying Q to contiguous d_qtmp buffer before normalization.
2. **MRoPE sections**: Host-side `precompute_rotary_host` didn't implement section frequency reset [11,11,10,0]. Fixed to match GPU `precompute_rotary_kernel`.
3. **Gate stride**: Chunked attention expected contiguous gate (stride q_dim), but fused buffer has stride q_dim_x2. Fixed by `copy_gate_from_fused_kernel` overwriting Q area.

### Files Created/Modified
- `src/wubu_model_gpu.cu` — NEW: GPU context management, GQA forward, RoPE table, cleanup
- `include/cuda_kernels.h` — Added `wubu_cuda_copy_q_from_fused`, `wubu_cuda_copy_gate_from_fused`
- `src/cuda_kernels.cu` — Added wrapper implementations for copy helpers
- `include/wubu_model.h` — Added `void *gpu_ctx` field + GPU function declarations
- `src/wubu_model.c` — Added `#ifdef GPU_SUPPORT` GPU GQA path + `wubu_model_gpu_free()` call
- `tools/gen_text.c` — Updated GPU init to call `wubu_model_gpu_init()` + CPU stubs
- `Makefile` — Added `wubu_model_gpu.o` compile rule + link to `gen_text_gpu`

### Build Commands
```
make gen_text_gpu                    # GPU-support binary
GPU=1 ./gen_text_gpu "Hello" 32      # GPU GQA + CPU SSM/MoE + GPU out proj
./gen_text "Hello" 32                # CPU-only (unchanged)
```

### Next: Phase 16 — GPU SSM Matmuls
- Need quantized GPU matmul kernel for Q5_K/Q6_K (like Q4_K out proj kernel)
- Keep SSM weights on GPU in native Q5_K/Q6_K format (~255MB for 30 layers)
- Wire `wubu_cuda_ssm_forward()` or `gpu_ssm_forward()` into forward pass
- Keep persistent SSM states on GPU
- Target: SSM from ~20ms → ~2ms per decode
