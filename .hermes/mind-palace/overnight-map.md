# WuBuText AI — Overnight Navigation Map

**Purpose:** This is the entry doc for long-running autonomous sessions. Read this, then fan out into the mind palace based on what's most actionable. No bootstrapping from scratch — the map tells you where to dig.

---

## Quick Trunk Reference

| What | Where |
|------|-------|
| Platform target map | `.hermes/references/qwen36_target_map.md` |
| Master plan (phases) | `.hermes/mind-palace/plans/master_impl_plan_v2.md` |
| Phase 3 detail | `.hermes/mind-palace/tier3-impl/10-training-loop/README.md` |
| Architecture (GGUF→CUDA→TST) | `.hermes/presentation/3-architecture.md` |
| Per-phase status | `.hermes/presentation/4-implementation-status.md` |
| Current state log | `.hermes/session-end-2026-05-13.md` |
| Fresh start prompt | `.hermes/mind-palace/fresh_start_prompt.md` |

---

## Where We Are (May 13)

**Phases 0-2 complete. Phase 3 active — multiple workstreams in progress.**

**Done — VERIFIED:**
- All 40 layers (30 SSM + 10 GQA) in C + CUDA — benchmarked 39x GPU vs CPU
- BBPE tokenizer (931 lines C) — matches HF for ASCII, **CJK merge hash bug NOT fixed**
- Complete platform target map validated against HF configs + GGUF binary (26KB)
- **Parallel associative scan CUDA kernel** — written at cuda_kernels.cu:399, NOT yet tested against CPU
- **Fused SSM forward** — written at cuda_kernels.cu:535, NOT yet tested against CPU
- **Train pipeline scaffold** — `train_real.c` loads model + tokenizer, runs forward pass, reports throughput (0.2 tok/s CPU). CE loss commented out (output weight disabled). Not yet training.
- **Corpus tokenizer** — `tokenize_corpus.c` + `extract_corpus.py` — extracted 5K narratives → 4.4MB → 320K tokens → `data/train_data.bin`
- **MoE weight reader** — `test_moe.c` loads all 8 MoE tensors and checks dimensions. Router + SwiGLU compute written but **NOT tested against reference** because IQ2 quant dequant produces garbage on this hardware. Computational structure is structurally correct but unverified by output comparison.
- 905 lines cuda_kernels.cu (+291 from prior session) — 24 kernels
- 13 new/updated tools and source files

**CLAIMS MARKED AS UNVERIFIED:**
- Stream E (MMProj) — NO dedicated tool exists. Dimensions were asserted from session summary but no C code reads the MMProj GGUF.
- "Parallel scan verified bit-exact" — need to see test_parallel_scan output to confirm
- "MoE computational structure verified correct" — loads weights and has the code, but IQ2 dequant gives garbage so no numerical verification against reference

**Active workstreams:**

### A — Parallel Scan Kernel ✅ WRITTEN, needs correctness test
`ssm_parallel_scan_kernel` at cuda_kernels.cu:399 — Blelloch-style prefix scan.
`test_parallel_scan.c` (376 lines) exists. Need: `make test_parallel_scan && ./test_parallel_scan` to verify against CPU.

### B — Tokenizer CJK Bug 🟡 IN PROGRESS
Merge(163,124) not in hash table. The pre-tokenizer produces correct single pre-token for Chinese, but byte-level merges fail because `find_merge()` returns -1 for the first CJK byte pair. Suspect: `merge_hash_key()` function or `build_merge_hash()` insertion bug.

### C — Training Pipeline 🟡 NEEDS DATA
`train_real.c` loads model + tokenizer + corpus and runs forward+CE loss. But needs:
1. Tokenized binary training data (from tokenize_corpus tool)
2. CJK tokenizer fix before tokenizing real Chinese text
3. AdamW integration with real model (not just finite-diff stub)

### D — MoE Forward Pass 🟡 IN PROGRESS
`test_moe.c` (334 lines) reads MoE weights from GGUF and verifies tensor dimensions. Router logic (top-8 of 256 experts) and SwiGLU computation need wiring.

### E — GQA Q Weight Dimension 🟡 NEEDS VERIFICATION
GQA Q weight [2048,8192] in GGUF but IQ2_M dequant may produce different number of elements. If dequant yields 4096, our code over-reads. Need: read, dequant, and count elements.

---

## Workstreams (pick one per session)

### Stream A — Parallel Scan Verification (blocker to unblock)
The kernel is written. Run `make test_parallel_scan && ./test_parallel_scan`. If it passes against CPU reference, the SSM training bottleneck is cleared. If it fails, debug the parallel scan kernel (cuda_kernels.cu:399).

### Stream B — Tokenizer CJK Merge Bug
Merge(163,124) not found in hash. Check `merge_hash_key()` function and `build_merge_hash()` insertion path. Compare hash key for (163,124) against what's expected from the GGUF merges list. Dump raw merges to verify the pair exists.

### Stream C — Tokenize a Corpus & Test Training
Once CJK tokenizer works: use `make tokenize_corpus && ./tokenize_corpus <text_file> <output.bin>`. Then `make train_real && ./train_real <model.gguf> <tokenized.bin>` to verify the pipeline.

### Stream D — MoE SwiGLU Forward Pass
`test_moe.c` reads weights. Wire the router (x @ ffn_gate_inp → softmax → top-8) and expert computation (silu(x @ gate) * (x @ up) @ down). Verify against CPU reference.

### Stream E — GQA Q Weight Dequant Count
Read GQA layer's `wq.weight` from GGUF, dequantize with IQ2_M, count elements. Compare to expected 4096 (or 8192). Fix `bench.c` weight load if over-reading.

### Stream F — MMProj Vision Merger (UNSTARTED)
No C code reads the MMProj GGUF yet. Need: read `mm.0[4608,4608] → GELU → mm.2[4608,2048]` from `/models/qwen3.6-35b-mmproj-F16.gguf`. Plus 27 ViT blocks (1152 hidden, 4304 FFN). Dimensions known from target map but no implementation exists.

### Stream G — Run Existing Tests and Report
Many tools were written but never run. `test_fused`, `test_fused_vs_old`, `test_parallel_scan`, `test_moe` — all exist in tools/ and Makefile but no runtime output is recorded. Build and run each, report pass/fail/actual diffs.

---

## Data You Should Not Re-Derive

---

## Data You Should Not Re-Derive

- R=0.956 for Poincaré ball (3 × mean_norm, verified 95% NN preservation)
- SSM recurrence: h[t] = h[t-1]·exp(gate) + K[t]·(V[t] − h[t-1]·K[t])·β[t]
- GQA Q weight is fused with gate: [2048,8192] = first 4096 Q + second 4096 gate
- GQA has SEPARATE q/k/v weights per layer (not fused QKV)
- attn_output_gate lives on SSM layers only (not GQA)
- MoE weights are 3D [2048, 512, 256], separate gate_exps/up_exps/down_exps in GGUF
- Vision activation: GELU tanh approximation, not SiLU
- Path: `/home/wubu/bytropix`. Models: `/models/`. HF configs: `/home/wubu/models/qwen36_og/`
- 23 source files, 7,066 lines total

---

## Fallback

If all streams look blocked, pick the one with the clearest next step and make a `plan.md` in `.hermes/plans/` for it. Better to document a plan than sit idle.
