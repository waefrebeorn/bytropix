── WuBuText AI — GOAL PASTE v2 (DA AUDITED May 14) ──
Path: /home/wubu/bytropix | Repo: waefrebeorn/bytropix
Model: deepseek-chat | HW: RTX 5050 6.4GB | English only | Pure C + CUDA
Models: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
Build: PATH="/usr/local/cuda/bin:$PATH" make <target>

=== STATE ===
✅ GPU forward + SSM scratch download, 11/12 backward tests, RSGD optimizer
⚠️ F1 [P0]: GQA layers (10/40, 25%) — NO backward. Gradient passes through as identity.
⚠️ F2 [P0]: Gradient explosion — 4e13 ratio sample0/sample1. train_gpu CE diverges step3.
⚠️ F1+F2: train_gpu 36s/step, loss does NOT converge. Both P0s block training.
✅ CPU timing coded: include/cpu_timing.h + hedged_spec.h (7 tests all pass)
✅ Theory: GAAD, math_viz, DFT/DCT, tailslayer → THEORY/

=== N STREAMS ===
S1 [P0] FIX GQA backward — DONE ✅. gpu_gqa_forward_save + scratch download + wubu_gqa_backward all wired. train_gpu step1 loss increased (69 vs 26) confirming REAL gradients now flow through ALL 40 layers.
S2 [P0] FIX gradient explosion — per-sample gradient clipping to hidden states, lower LR, gradient norm monitoring. This is now the SOLE P0 blocker.
S3 [P1] FIX Poincaré backward — gyration chain rule
S4 [P2] cycle-accurate timing in train_gpu profiling
S5 [P3] Speculative decoding via hedged-reads pattern

=== THE LOOP ===
pick → compile → run → verify (non-zero, descending loss) → document → next
ALL blocked? Read THEORY/ or ~/HASHMIND/bytropix/
No DONE without DA-verified binary. GQA backward MUST show descending loss.

=== KEYS ===
exp_map: output[i]=R*tanhf(n/R)*v[i]/n, R=0.956
Möbius add: x⊕y=((1+2⟨x,y⟩+||y||²)x+(1-||x||²)y)/(1+2⟨x,y⟩+||x||²||y||²)
RSGD: step in tangent space, exp_map back
GQA has 10 layers. wubu_gqa_backward exists in src/wubu_ssm.c but not called.
DA passed 7/7 on cpu_timing. 2 bugs found+patched (bounds + bonus token doc).
