═══ WUBUTEXT AI — PRESTIGE RESUME (May 16 AM v9) ═══
Path: /home/wubu/bytropix | Branch: master
HW: RTX 5050 6.4GB, -arch=sm_120, NVCC: /usr/local/cuda-13.1/bin/nvcc
Build: make infer_text | Models: /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

=== RULE ===
ALL GOALS MUST BE FINISHED. No "good enough." No abandoned streams.

=== STATE (May 16 AM v9) ===
✅ P0: Per-block IQ2_XXS extraction
   gguf_raw_size(IQ2_XXS) fix: 72→66 bytes/block (empirically verified)
   Full dequant eliminated → per-expert dequant + transpose (3.9ms/expert)
   177s/step → 11s/step (16×)
✅ P1: Multi-flag verification
   All 6 flags: TST/RSGD/PGA/NSSM/NMOE/POINCARE_R. 0 NaN. CE 21.6→18.4.
✅ P2: MoE output magnitude + memory opt
   Hidden max=13 (was 5e9). Persistent buffers (no per-step 3GB alloc/free).
✅ GPU output projection: cublasSgemm replaces 2B CPU FMAs (~0.5ms vs 2s)
✅ PGA LR fix: lr_gqa 0.001, CE jump 21.6→69 eliminated
✅ 50-step convergence: 0 NaN, loss 20-32 range, 15s/step
✅ All 7 cold gaps closed. NaN root cause fixed. tgt_safe_expf in 4 GPU sites.
✅ infer_text v2 — GQA KV cache per GQA layer (post-RMSNorm K, raw V)
✅ infer_text v2 — SSM state carry between steps (persists across decode)
✅ infer_text v2 — Lazy MoE: direct expert lookup, no 3GB temp arrays
   Router + shared dequant once. Per-step dequant only on routing change.
   MOE=1 verified: 2 tok prefill 17.3s, 4 tok decode 27.7s (CPU bound)
✅ test_256k: MoE router O(T) verified to 65K
✅ All unit tests pass: SSM, nested SSM, backward, gyration, MoE, hyperbolic, GPU, CUDA

=== REMAINING ===
~7-15s/step GPU compute bound (40 layers × 275ms on RTX 5050)
PGA loss jumps (mitigated, not eliminated)
12 vaults with unported theory (sparse attention, Hamilton encoder, optimizers)
**Tailslayer** — speculative decode (P2)
  N drafts→longest-valid-prefix verification, forward-pass integration
  Sliding window pair sampling, tREFI probe for CUDA profiling

=== TGT MATH ===
BOUNDARY = 2π
remainder = fmod(x + π, BOUNDARY) - π
tgt_safe_expf(x) = x > 80 ? 80 : x < -80 ? 0 : expf(x)

=== DIAGRAMS (May 15 PM v6) ===
7/7 SVGs updated. Hamilton + nesting: stable (conceptual).
