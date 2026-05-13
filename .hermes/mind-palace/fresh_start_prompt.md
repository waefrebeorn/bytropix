── WuBuText AI — FRESH START PROMPT — Paste this as your first message ──

You're working on WuBuText AI — building Qwen3.6-35B-A3B from scratch in pure C with WuBu hyperbolic geometry. Path: /home/wubu/bytropix.

== BOOT SEQUENCE ==
1. Load skill wubu-mind-palace with skill_view(name='wubu-mind-palace')
2. Read ~/bytropix/.hermes/mind-palace/README.md — nav table of contents
3. Read ~/bytropix/.hermes/mind-palace/plans/master_impl_plan_v2.md — the plan
4. Read ~/bytropix/.hermes/mind-palace/plans/devils_advocate_v2.md — risks
5. Read ~/bytropix/.hermes/mind-palace/tier3-impl/10-training-loop/README.md — Phase 3 details
6. Read ~/bytropix/.hermes/references/TST_TOKEN_SUPERPOSITION.md — Phase 3 training method
7. skim_session_search for recent wubu sessions (last 3) to see what was just done

== THE LOOP ==
For every session, do this cycle:
  read_plan → find_current_step → execute it → mark done in plan doc → report_next

Don't ask what to do. Read the plan, find the next undone step, do it, check it off, tell me the next one. If blocked, say "BLOCKED on X — need Y from you".

== KEY FACTS (don't re-derive these) ==
- Phase 1 is DONE. Embeddings at data/qwen36_embeddings_c.bin (2.03GB, Poincaré-mapped, R=0.956)
- Phase 2 is DONE: 30 SSM + 10 GQA layers in C + CUDA. All 40 layers verified.
- Phase 2.5 GPU test DONE: 9.53 tok/s GPU vs 0.20 tok/s CPU = 47.83x speedup
- CUDA infra: src/cuda_kernels.cu, include/cuda_kernels.h, bench tool at tools/bench_e2e.c
- Phase 3 now: training loop — using Token-Superposition Training (TST) method
- TST paper: .hermes/references/TST_TOKEN_SUPERPOSITION.md (also printed PDF)
- TST: bag s tokens → avg embeddings → forward on L/s → MCE loss → recovery with standard CE
- Poincaré R = 0.956 = 3×mean_norm (from Phase 1 analysis)
- Tokenizer BBPE fixes needed: O(N²) merge lookup (linear scan → hash table)
- C only for production. Python only for prototypes/analysis.
- Working dir: /home/wubu/bytropix. Model weights: /mnt/wslg/distro/models/
- English only. No Chinese.

== WHERE WE ARE ==
Phase 3 — Training Loop. First: fix tokenizer O(N²) merge, implement encode/decode, then implement TST loss (bag embeddings + MCE), then stub training loop with gradient descent.

== THE GOAL ==
Keep making forward progress. Each session does the next step. No time estimates. No summaries unless things went wrong. Just ship the next piece of C code, verify it compiles and runs, check it off, and tell me what comes next.
