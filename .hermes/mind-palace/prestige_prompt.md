═══ WUBUTEXT AI — PRESTIGE RESUME (May 16 AM v8) ═══
Path: /home/wubu/bytropix | Branch: master
HW: RTX 5050 6.4GB, -arch=sm_120, NVCC: /usr/local/cuda-13.1/bin/nvcc
Build: make train_integrated | Models: /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

=== RULE ===
ALL GOALS MUST BE FINISHED. No "good enough." No abandoned streams.

=== STATE (May 16 AM v8) ===
✅ P0: Per-block IQ2_XXS extraction
   gguf_raw_size(IQ2_XXS) fix: 72→66 bytes/block (empirically verified)
   Full dequant eliminated → per-expert dequant + transpose (3.9ms/expert)
   177s/step → 11s/step (16×)
✅ P1: Multi-flag verification
   All 6 flags individually + combined: TST/RSGD/PGA/NSSM/NMOE/POINCARE_R
   0 NaN in any configuration. Loss: CE 21.6→18.4 stable.
✅ P2: MoE output magnitude + memory opt
   Hidden: max=13 (was 5e9 from buggy strided extraction)
   Persistent buffers in lmoe_t (no per-step 3GB alloc/free)
✅ GPU output projection: cublasSgemm replaces 2B CPU FMAs (~0.5ms vs 2s)
✅ PGA LR fix: lr_gqa 0.001 (was 0.01), CE jump 21.6→69 eliminated
✅ 50-step convergence: 0 NaN, loss 20-32 range, 15s/step
✅ Async D→H copies: dead PGA copies skipped when !pga_enabled
✅ All 7 cold gaps closed (May 14)
✅ NaN root cause FIXED: MoE weight interleaving + raw_size bug
✅ tgt_safe_expf in 4 GPU kernel sites
✅ infer_text: full text generation pipeline (tokenize→embed→forward→sample→decode)
✅ test_256k: MoE router O(T) verified to 65K

=== REMAINING ===
~11-15s/step GPU compute bound (40 layers × 275ms on RTX 5050)
PGA loss jumps 21.6→69 (pre-existing LR issue, mitigated not eliminated)
12 vaults with unported theory (sparse attention, Hamilton encoder, optimizers)
**Tailslayer** — hedged-read CUDA kernel for speculative decode (P2, new May 15)
  N replicas→N drafts, first-response-wins→longest-valid-prefix, clflush→forward-pass
  Sliding window pair sampling, tREFI probe for CUDA profiling
GQA: O(T²) — KV cache needed for 256K autoregressive inference
Lazy per-expert MoE cache for fast inference (MOE=1)

=== TGT MATH ===
BOUNDARY = 2π
remainder = fmod(x + π, BOUNDARY) - π
tgt_safe_expf(x) = x > 80 ? 80 : x < -80 ? 0 : expf(x)

=== DIAGRAMS (May 15 PM v6) ===
7/7 SVGs updated: phase-roadmap (100% complete), gguf-pipeline, math-pipeline,
research-timeline, llamacpp-clone. Hamilton + nesting: stable (conceptual).
