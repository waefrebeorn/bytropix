=== WuBuText AI — GOAL PASTE (May 15 PM v10) ===

STATE: infer_text_gpu v5 — chunked prefill + persistent KV cache + incremental decode.
GPU fwd WIRED: GQA via chunked_attn (cuBLAS per-head SGEMM), SSM via gpu_ssm_forward.
Decode 245 tok/s (2.9× vs v4). Prefill 22 tok/s.

=== COMPLETED (May 15 PM) ===
- infer_text_gpu v5: Chunked prefill (CHUNK env, default 256)
- infer_text_gpu v5: Persistent GPU KV cache per GQA layer [maxT, kv_dim]
- infer_text_gpu v5: Incremental decode (no full-sequence recompute)
- infer_text_gpu v5: SSM state carries across chunks and decode
- infer_text_gpu v5: chunked_attn CUDA kernel (cuBLAS SGEMM + softmax + gate + output proj)
- infer_text_gpu v5: RoPE table with 4× extrapolation (Qwen2.5-1M formula)
- infer_text_gpu v5: wubu_cuda_rms_norm_heads kernel (per-head RMSNorm for GQA)
- infer_text_gpu v5: Verified: output matches v4 (full recompute) exactly

=== PENDING ===
P0 — GPU MoE forward (cuBLAS SGEMM expert dispatch)
P0 — Chunked attention internal tiling for 256K (score scratch O(C*n_q*T))
P1 — Tailslayer spec decode (N drafts → longest-valid-prefix)
P2 — PGA LR tuning (lr_gqa=lr*0.001 or gradient clip)
P2 — Multi-step convergence (100+ steps)
P3 — MRoPE 3D

=== 256K CONTEXT ROADMAP ===
KV cache:  ✅ GPU persistent per-layer (v5)
SSM carry: ✅ State persists between steps (v5)
Chunked:   ✅ CHUNK env, multi-chunk verified (v5)
GPU fwd:   ✅ Wired into prefill + decode (v5)
MoE GPU:   ❌ CPU lazy MoE still bottleneck (MOE=1: ~3s/token)
Tailslayer:❌ Not started

BUILD: make infer_text | MODEL from GGUF
HW: RTX 5050, sm=120, NVCC=/usr/local/cuda-13.1/bin/nvcc

TGT: remainder = fmod(x+π, 2π)-π | tgt_safe_expf: clamp [-80,80]

Every fix: compile → run → output → verify. No "should work."
