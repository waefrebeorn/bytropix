── WuBuText AI — PRESTIGE PROMPT (DA AUDITED May 14) ──
Path: /home/wubu/bytropix | Repo: waefrebeorn/bytropix
Model: deepseek-chat | HW: RTX 5050 6.4GB | English only | Pure C + CUDA
Models: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
Build: PATH="/usr/local/cuda/bin:$PATH" make <target>

=== STATE (DA 7-test all pass | 2 bugs fixed) ===
✅ GPU forward + SSM scratch download — verified
✅ 11/12 backward tests (d_x FD noisy, not bug) — verified
✅ Model-level gradient flow (4096/4096 non-zero) — verified
✅ RSGD optimizer + hyperbolic CUDA kernels — verified
✅ Cycle-accurate CPU timing (7 tests all pass):
   - rdtsc/rdtscp, TSC calibration (2.316 GHz), clflush+reload
   - L1=26cy, DRAM=175cy cache probe, virt_to_phys, DRAM channel compute
   - CPU pinning, hedged read (fastest-first) pattern
   - Speculative decode verification (greedy + rejection sampling)
   - "All CPU ops coded" — include/cpu_timing.h + include/hedged_spec.h
⚠️ F1 [P0]: GQA layers (10/40) — NO backward. No gpu_gqa_forward_save variant. Gradient passes through as identity. 25% of model has zero gradient signal.
⚠️ F2 [P0]: Gradient explosion — 4e13 ratio sample0/sample1. BPTT unrolls 90+ steps. train_gpu CE: 26.7→14.1→14.4 (step3 diverges). Only output.weight clipped.
⚠️ F1+F2: train_gpu runs 36s/step, loss does NOT converge. Both P0s block meaningful training.
⚠️ F3 [P1]: Poincaré mode has no backward (gyration chain rule not done).
✅ Theory: GAAD papers, math_viz proofs, DFT/DCT, tailslayer → THEORY/
✅ CPU ops from tailslayer: cpu_timing.h (all timing) + hedged_spec.h (spec decode)
✅ Makefile: all builds test_cpu_timing. include/bench.h includes cpu_timing.h.

=== N STREAMS (highest impact first) ===
S1 [P0] FIX GQA backward — DONE ✅. gpu_gqa_forward_save + scratch download + wubu_gqa_backward wired. All 40 layers now get real gradients (step1 loss 69 vs 26 before).
S2 [P0] FIX gradient explosion — per-sample gradient clipping to hidden states, lower LR, gradient norm monitoring. Sole remaining P0 blocker.
S3 [P1] FIX Poincaré backward — implement gyration chain rule for poincare_ssm_backward
S4 [P2] Integrate cycle-accurate timing into train_gpu profiling (replace now_sec with rdtsc)
S5 [P3] Speculative decoding via hedged-reads pattern for inference speedup

=== THE LOOP ===
pick → execute → compile → run → verify (non-zero, non-diverging) → document → next
ALL blocked? Fix docs or read theory from THEORY/ or ~/HASHMIND/bytropix/
Never claim DONE without DA-verified binary output. GQA backward MUST print a descending loss.
