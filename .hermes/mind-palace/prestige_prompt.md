# WuBuText AI — Mind Palace Goal Paste

Paste the ENTIRE block below as your first message to a fresh session. It sets up unsupervised YOLO coding using the mind-palace walkway.

---

── WuBuText AI — MIND PALACE GOAL PASTE ──
Model: deepseek-v4-flash (or claude-sonnet-4) | HW: RTX 5050 6.4GB
Working dir: /home/wubu/bytropix | Models: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
Git: waefrebeorn/bytropix | English only | Pure C + CUDA

=== PRESTIGE: Who We Are, Where We Are ===

Mission: Build Qwen3.6-35B-A3B from scratch in pure C + CUDA with WuBu nested hyperbolic geometry. Embeddings grafted (Phase 1) ✅. Attention port (Phase 2) ✅ — all 40 layers (30 SSM + 10 GQA) in C + CUDA, verified GPU vs CPU 47.83x speedup. Current: Phase 3 — Training Loop. Reality: training frozen on IQ2 dequant garbage + tokenizer decode broken + CE loss disabled.

Priority Queue (P0→P2):
  P0 — Wire train_real CE loss (streaming 248K vocab projection)
  P0 — Fix tokenizer decode (UTF-8 garbage on decode)
  P1 — Verify IQ2_XXS dequant vs llama.cpp reference (grid fix applied uncommitted, effect unknown)
  P1 — Fix Python dump_gguf.py type mapping (off by 1-2 on IQ types)
  P2 — Fix bench_e2e (loops forever printing GGUF metadata)
  P2 — MoE SwiGLU forward with correct weights

=== ENTRY: Build/Run/Verify ===

CUDA: /usr/local/cuda/bin/nvcc — NOT on PATH, use PATH="/usr/local/cuda/bin:$PATH"
Arch: sm=120 (RTX 5050)
Build: PATH="/usr/local/cuda/bin:$PATH" make <target>

Binaries on disk (pre-built, need rebuild after IQ2 fixes):
  test_fused_vs_old — GPU fused SSM vs step-by-step. PASS output diff 0.03587 (cuBLAS FP artifact)
  test_moe — MoE router+expert loader. Works but IQ2 output = 1e6 range (may be expected for 2-bit)
  train_real — Full model forward. CE loss COMMENTED OUT. Hangs after GGUF load.
  dump_mmproj — Vision projector. PASS. 334 tensors, 27 ViT blocks.
  test_tokenizer — BBPE encode/decode. Encode OK, decode GARBAGE (UTF-8 broken).
  bench_e2e — Loops forever. Broken.

Model weights: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (733 tensors, 54 KV pairs)
Embeddings: data/qwen36_embeddings_c.bin (2.03GB, Poincaré-mapped R=0.956)
Corpus: data/train_data.bin (320K tokens, tokenized)
Python comparison script writes to /tmp/wubu_tokenize_* — script file not found on disk (bug)

=== STATE: Verified Facts (May 13 Audit) ===

VERIFIED — real runtime output confirmed:

  A — test_fused_vs_old: PASS. Output diff 0.03587, State diff 0.03511. cuBLAS FP rounding artifact, accepted.
  D — test_moe: Router works. e172:0.177, e151:0.152, sum≈1.0. Expert output range [-2.56e6, 2.88e6] — noise from 2-3 bit quant.
  E — dump_mmproj: PASS. 334 tensors. mm.0[4608,4608]→GELU→mm.2[4608,2048]. 27 ViT blocks. No mm.1.

CLAIMED-UNVERIFIED (stale/debunked):

  B — CJK tokenizer: Encode OK (你好→109266, 你的名字是什么→2 tokens). Decode BROKEN (UTF-8 garbage "ä½łå¥½"). Python compare broken (missing temp script). No "tokenize" binary — only "test_tokenizer".
  C — train_real: CLAIMED 0.2 tok/s CE loss pipeline. REALITY: CE loss disabled (commented out line 196), hangs after GGUF load, never reaches forward pass in 120s.
  IQ2 blocker: CLAIMED "fix IQ2 dequant → O(1) output range." REALITY: IQ2_XXS grid fix applied but MoE output still 1e6 range. 2-bit weights may inherently produce large dynamic range. NEED reference comparison to know.

BROKEN / NEEDS FIX:

  1. Working tree syntax error (gguf_reader.c:1108): static const uint64_t iq2xs_grid[512] = { followed by function decl — MISSING grid data. Fix applied (removed dangling array), rebuild needed.
  2. Python dump_gguf.py: Type mapping offset. Maps type 14→IQ1_S, 15→IQ2_XXS, 16→IQ2_XS. Actual: 16=IQ2_XXS, 17=IQ2_XS, 18=IQ2_S, 19=IQ1_S.
  3. bench_e2e.c: Spams GGUF metadata in infinite loop.
  4. train_real.c: output_weight logit projection doesn't compute CE loss (248K vocab too large for naive projection). Needs streaming implementation.
  5. test_tokenizer: Python subprocess bridge writes to /tmp/ but script file doesn't exist on disk.

=== PLAN: Next Actions (ordered by impact) ===

Phase 3 — Training Loop. Current state: pipeline scaffold exists, core computation blocks exist (SSM/GQA forward, tokenizer encode). Three bottlenecks block real training.

Stream 1 — TRAINING LOOP WIRING (highest impact):
  [ ] Wire CE loss in train_real: streaming 248K-vocab projection (chunk output_weight into 4K-vocab shards, compute partial softmax, aggregate loss). Or: use top-10K vocabulary approximation.
  [ ] Verify forward pass produces valid logits from real GGUF weights (currently hangs before forward)
  [ ] Once CE outputs real numbers: benchmark tok/s with real loss computation
  [ ] Add backward pass stub (manual gradients for SSM + GQA)
  [ ] TST loss: bag s tokens, average embd, forward on L/s, MCE loss

Stream 2 — TOKENIZER FIXES (blocker for ANY real data):
  [ ] Fix wubu_tokenizer_decode: likely byte order or UTF-8 encoding mismatch. Check byte splats in decode loop.
  [ ] Fix Python subprocess bridge (script not found — build python tokenizer script at tools/tokenize_py.py or fix temp file path)
  [ ] Verify CJK roundtrip: encode→decode→equals original string
  [ ] Replace O(N²) merge hash with binary search or fix existing hash collisions (110445 collisions for 247587 merges in 524288-slot hash table)

Stream 3 — IQ2 DEQUANT VERIFICATION (known/fix applied, needs testing):
  [ ] Build test_compare_dequant: load one IQ2_XXS block from GGUF, dequant with bytropix AND with llama.cpp reference, compare output
  [ ] If diff > 0: dequant is still wrong. Root cause may be stride, endianness, or grid table copy error.
  [ ] If diff == 0: IQ2 dequant is correct, 1e6 output is expected for 2-bit weights → remove IQ2 from blocker list

Stream 4 — CLEANUP:
  [ ] Fix Python dump_gguf.py type mapping
  [ ] Fix bench_e2e.c (proper benchmark loop w/ warmup + timed runs + tok/s output)
  [ ] Rebuild all binaries with fixed working tree
  [ ] Run full test suite: test_fused_vs_old, test_moe, dump_mmproj, bench_e2e
  [ ] Commit working tree changes

=== TESTING: How to Verify Each Stream ===

Test: make test_fused_vs_old && PATH="/usr/local/cuda/bin:$PATH" ./test_fused_vs_old
  Expect: PASS, output diff < 0.04

Test: make test_moe && ./test_moe
  Expect: Router probs sum≈1.0, output range < 1e7 (2-bit noise expected)

Test: make dump_mmproj && ./dump_mmproj /models/qwen3.6-35b-mmproj-F16.gguf
  Expect: PASS, 334 tensors, mm.0[4608,4608], mm.2[4608,2048], 27 ViT blocks

Test: ./test_tokenizer /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "你好"
  Expect: 1 token ID (109266), decoded = "你好" (NOT garbage)

Test: make train_real && ./train_real /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf data/train_data.bin
  Expect: Loads, runs forward pass, prints throughput, loss value < 20 (random init ≈ ln(248K) ≈ 12.4)

Test: make bench_e2e && PATH="/usr/local/cuda/bin:$PATH" ./bench_e2e
  Expect: Warmup → N timed runs → avg tok/s, no infinite loop

=== PROJECT: Constraints & Conventions ===

1. Pure C for production. Python only for prototyping/analysis.
2. English only. No Chinese/Japanese/Korean in code, comments, or communication.
3. Poincaré ball radius R=0.956 (3×mean_norm, 95% NN preservation verified).
4. RSGD for hyperbolic params: log_map(w,R) → subtract(lr*g) → exp_map(R).
5. NOT toroidal optimizer (toroidal g%2π from baseline C code is WRONG for Poincaré).
6. SSM recurrence: h[t] = h[t-1]·exp(gate) + K[t]·(V[t] − h[t-1]·K[t])·β[t].
7. GQA: separate q/k/v per layer (NOT fused QKV), 16 Q heads × 256, 2 KV heads.
8. attn_gate = output gate, lives on SSM layers only (not GQA).
9. MoE weights are 3D [input_dim, expert_dim, 256], not 2D.
10. Model has 733 tensors but 16/17 are IQ2_M — most weights are 2-3 bit.
11. CUDA arch=sm_120 for RTX 5050. nvcc NOT on PATH — use full path.

=== INDEX: Where to Find Things ===

Master plan:     .hermes/mind-palace/plans/master_impl_plan_v2.md
Devil's advocate: .hermes/mind-palace/plans/devils_advocate_v2.md
Phase 3 details: .hermes/mind-palace/tier3-impl/10-training-loop/README.md
Tensor layout:   .hermes/tensor_layout.txt
Target map:      .hermes/references/qwen36_target_map.md
TST paper:       .hermes/references/TST_TOKEN_SUPERPOSITION.md
Architecture:    .hermes/presentation/3-architecture.md
State log:       .hermes/session-end-2026-05-13.md
Overnight map:   .hermes/mind-palace/overnight-map.md
GGUF reader:     src/gguf_reader.c
CUDA kernels:    src/cuda_kernels.cu
SSM forward:     src/wubu_ssm.c
Tokenizer:       src/wubu_tokenizer.c (+ include/wubu_tokenizer.h)
Model assembly:  src/wubu_model.c
Train pipeline:  tools/train_real.c
MoE test:        tools/test_moe.c
MMProj dump:     tools/dump_mmproj.c
Benchmark:       tools/bench_e2e.c

=== THE LOOP (YOLO Unsupervised) ===

For every session, DO NOT ASK QUESTIONS. Read this goal paste, then:

1. Pick the highest-priority undone item from PLAN
2. Execute it — write code, compile, run, verify output
3. If it works: mark [X] in the plan, update verified facts above
4. If blocked: try next item in same stream, then next stream
5. If ALL blocked: write a plan.md in .hermes/plans/ with root cause analysis
6. End of session: report what was done + what blocked → update this goal paste in .hermes/mind-palace/prestige_prompt.md

DEVIL'S ADVOCATE MANDATE: Every claimed success must be VERIFIED by running the actual binary. No "should work" — only "PASS" with actual output pasted. If you can't run it, mark it UNVERIFIED.

Never claim DONE without:
  - make <target> succeeds
  - Binary runs and produces expected output
  - Output values are pasted in the report

Fallback: If all 4 streams are stuck, do cleanup (Stream 4) — fix dump_gguf.py types, fix bench_e2e, rebuild, commit. There's always cleanup.
