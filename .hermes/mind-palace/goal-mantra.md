=== WuBuText AI — GOAL PASTE (May 16 AM v9) ===
ALL GOALS MUST BE FINISHED.

STATE: infer_text v2 with GQA KV cache + lazy MoE + SSM state carry.
256K decode feasible via KV cache (O(T) instead of O(T²)).
MOE=1 works: 2 tok prefill 17.3s, 4 tok decode 27.7s (CPU-bound).

=== COMPLETED (May 16) ===
- infer_text v2: GQA KV cache per-layer (post-RMSNorm K, raw V)
- infer_text v2: SSM state carry between steps (no reset per iteration)
- infer_text v2: Lazy MoE cache — direct expert lookup, no 3GB temp arrays
  - Router + shared expert dequant once at startup
  - Only dequantizes experts when routing changes between steps
  - Per-layer cached experts: ~96MB per active layer
- infer_text v2: Online softmax fallback for T > 64K contexts
- infer_text v2: Two-phase pipeline (prefill + decode)

=== PENDING ===
P1 — GPU forward acceleration for decode (gpu_gqa_forward, gpu_ssm_forward exist)
P1 — Tailslayer spec decode (N drafts → longest-valid-prefix)
P2 — PGA LR tuning (lr_gqa=lr*0.001 or gradient clip)
P2 — Multi-step convergence (100+ steps)
P3 — MRoPE 3D

=== 256K CONTEXT ROADMAP ===
KV cache:  ✅ GQA per-layer append-only cache
SSM carry: ✅ State persists between steps
MoE cache: ✅ Lazy per-expert, no 3GB arrays
GPU fwd:   ❌ Has kernels, not wired into decode loop
Tailslayer:❌ Not started

BUILD: make infer_text | MODEL from GGUF
HW: RTX 5050, sm=120, NVCC=/usr/local/cuda-13.1/bin/nvcc

TGT: remainder = fmod(x+π, 2π)-π | tgt_safe_expf: clamp [-80,80]

Every fix: compile → run → output → verify. No "should work."
