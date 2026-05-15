═══ WUBUTEXT AI — GOAL PASTE (May 15 PM v6) ═══
ALL GOALS MUST BE FINISHED.

STATE: P0-P2 ALL DONE. Training 177s→11s/step (16×). Loss CE 21.6→18.4, 0 NaN.
gguf_raw_size(IQ2_XXS) fix: 72→66. Per-expert dequant (3.9ms/expert).
GPU output projection: cublasSgemm. All 6 flags verified + combined, 0 NaN.
Hidden max=13 (was 5e9 from buggy strided extraction). Persistent lmoe_t buffers.

=== ROADMAP (v6 — comprehensive, +tailslayer May 15) ===

P0 — GPU MoE forward (upload 96MB active experts → cuBLAS, eliminate 40 syncs)
P1 — PGA LR tuning (lr_gqa=lr*0.01→lr*0.001, stop CE jump 21.6→69)
P1 — Multi-step convergence (50+ steps, CE < 5.0)
P2 — MRoPE 3D (section=[11,11,10] missing from RoPE impl)
P2 — Architecture verify (KV heads=2, head_dim=256/128, conv_kernel=4, theta=10M)
P2 — Sparse attention port (O(n·k) linear, highest ROI vault port)
P2 — Tailslayer spec-decode kernel (hedged-read CUDA: N draft verify, first-valid-wins)
P2 — Sliding window pair sampling for draft-target logit alignment
P2 — Q-Controller + PID port (tiny JAX → C, reusable optimizer)

=== TAILSLAYER FINDINGS (May 15) ===
Source: LaurieWired/tailslayer — hedged reads across independent DRAM channels.
4 files in THEORY/papers/tailslayer-*. Direct matches:
  - N-replica→N-draft (speculative decode kernel P2)
  - clflush+reload → forward pass timing
  - First-response-wins → longest valid prefix accept
  - Channel bit → shared memory bank conflict analysis (P3)
  - tREFI probe → CUDA kernel profiling (P3)
  - Sliding window pair sampling → draft-target time alignment (P2)
  - N-way N≤channels → MoE SM dispatch scaler (P3)

=== PAPER AUDIT (32 Qwen files) ===
Discrepancies found: KV heads=2 (not 4), head_dim 256/128 split, MRoPE missing,
MTP head missing, bos=eos=248044 (same token). See plan.md for full table.

=== VAULT AUDIT (12 vaults) ===
Highest ROI: Sparse Attention (PyTorch), Q-Controller+PID (JAX), Hamilton Encoder (CUDA).
Low priority: Diffusion, Audio, Phase3, draftPY.

BUILD: make train_integrated
MODEL: /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
HW: RTX 5050, sm=120, NVCC=/usr/local/cuda-13.1/bin/nvcc

TGT: remainder = fmod(x+π, 2π)-π | tgt_safe_expf: clamp [-80,80]

Every fix: compile → run → output → verify. No "should work."
