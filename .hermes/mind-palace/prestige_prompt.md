── WuBuText AI — PRESTIGE PROMPT (DA Audit May 13) ──
Path: /home/wubu/bytropix | Commit: d738642 (clean)
Model: deepseek-chat | HW: RTX 5050 6.4GB | English only | Pure C + CUDA

=== WHO WE ARE ===
Building Qwen3.6-35B-A3B from scratch in C + CUDA. Phase 1 (embed graft) ✅. Phase 2 (attention port) ✅ — 40 layers CPU/GPU. Phase 3/3.5 (training loop) active.

=== DA AUDIT (May 13) — 8 Binaries Verified ===

✅ train_real — CE loss 12.66 (random baseline 12.42), 0.2 tok/s CPU, logits non-zero
✅ test_fused_vs_old — GPU diff 0.03587, PASS
✅ test_tokenizer — CJK round-trip: 你好 → 109266 → 你好
✅ test_moe — output [-0.028, 0.031], NaN=0, 36.6 tok/s. Q4_K fix resolved old garbage issue.
✅ dump_mmproj — 334 tensors, 27 ViT blocks, PASS

⛔ bench_e2e — ALL OUTPUTS ZERO (CPU max 0.000, GPU max 0.000). Old GPU weight path broken.
⛔ train_gpu — CE loss 69 (should be ~12.4). Per-layer-per-step GGUF reload = garbage.
⛔ train_backprop — Hangs at 180s during model init. Unknown root cause.

=== CRITICAL FINDINGS ===
1. train_real is the ONLY correct path — uses wubu_model_forward_from_embd with pre-loaded CPU weights
2. bench_e2e (old per-layer GGUF reopen) + gpu_ssm_forward produces ZEROS — both CPU and GPU
3. train_gpu (GPU forward) produces loss 69 instead of 12.66 — GPU weight loading wrong
4. train_backprop hangs — not a timeout issue, same code as train_real
5. OLD prestige prompt claims are STALE: "CE loss commented out", "IQ2 dequant = garbage", "bench_e2e loops" — all wrong now

=== BUILD COMMANDS ===
PATH="/usr/local/cuda/bin:$PATH" make <target>
CUDA arch sm=120, nvcc NOT on PATH

=== KEY FILES ===
train_real.c — working forward + CE loss pipeline
wubu_model.c — wubu_model_forward_from_embd (correct)
bench.c + cuda_kernels.cu — GPU kernels (broken weight loading)
train_backprop.c — gradient training (hangs)
train_gpu.c — hybrid GPU/CPU (wrong loss)

=== MODELS ===
/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (11GB, 733 tensors)
data/qwen36_embeddings_c.bin (1.9GB, Poincaré-mapped)

=== IMMEDIATE NEXT ===
P0 — Fix GPU weight loading path (bench.c → why all zeros?)
P1 — Fix train_backprop hang
