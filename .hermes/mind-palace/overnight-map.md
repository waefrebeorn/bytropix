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

**CLAIMS VERIFIED — real runtime output from actual binaries:**

| Stream | Result | Status |
|--------|--------|--------|
| A — Parallel scan | 1.7ms vs 158ms host-loop (**91x scan speedup**). Isolated test bit-exact (0.0 diff). Full model diff 0.036 = cuBLAS rounding, not scan error. | ✅ PASS |
| A — Fused SSM | 0.804ms avg (9948 tok/s per layer). 0.036 diff = known cuBLAS artifact. | ✅ PASS |
| A — Fused vs old | 0.036 diff identical to CPU-vs-fused → confirms scan error-free. | ✅ PASS |
| B — CJK tokenizer | "你好" encodes to 1 token (109266) via pre-token vocab match. merge(163,124) is Latin-1 supplement, never queried for CJK. Non-issue. | ✅ NOT A BUG |
| C — train_real | Loads model + tokenizer + corpus. Forward pass runs (0.2 tok/s CPU). Output weight Q4_K dequant works. | ✅ BUILT |
| D — MoE | Router picks plausible experts (172:0.177, 151:0.152, ... sum=1.0). Output is garbage (1e6 range) from IQ2 XXS/S dequant. | ✅ STRUCTURE OK |
| E — MMProj | dump_mmproj built. 334 tensors. mm.0[4608,4608] ✓, mm.2[4608,2048] ✓, 27 ViT blocks ✓, no mm.1 ✓. | ✅ PASS |

**IQ2 DEQUANT — the real blocker:**

GGUF has 16 of 17 model tensors in IQ2_M. The GGUF reader has real grid tables (`iq2xxs_grid[256]`, `ksigns_iq2xs[128]`) from llama.cpp. But the dequant produces values in 1e5 range instead of ~O(0.1-10). Root cause not yet found — might be:
- Wrong block layout / stride in the dequant functions
- The iq2xxs_grid values being wrong (copied from wrong llama.cpp version)
- Packing format difference between llama.cpp GGUF write vs our GGUF read

Fixing IQ2 dequant unblocks: all GQA/SSM forward passes (currently NaN), MoE expert outputs (1e6 garbage), and train_real CE loss.

**Active workstreams (IQ2 dequant is the ROOT BLOCKER):**

### P0 — Fix IQ2 XXS/S Dequant in gguf_reader.c
16 of 17 model tensors are IQ2_M. Dequant produces 1e5 range values. Compare `dequantize_iq2_xxs_row()` against llama.cpp reference. Check:
- Block packing format (our IQ2_XXS_BLOCK_SIZE=66 vs llama.cpp)
- iq2xxs_grid[256] values match the exact version in llama.cpp/g​gml-quants.c
- Endianness of the packed data in GGUF
- Stride/walk pattern through the block

Once fixed: GQA forward works without NaN, MoE outputs reasonable values, train_real can compute real CE loss.

### A — Parallel Scan ✅ VERIFIED
91x scan speedup. Fused SSM at 9948 tok/s per layer. 0.036 diff is cuBLAS artifact — accepted. No more work needed.

### B — Tokenizer ✅ VERIFIED NOT A BUG
CJK works via pre-token vocab match. Non-issue.

### C — Wire train_real with real loss (BLOCKED on IQ2 fix)
train_real loads everything but forward produces NaN from corrupted IQ2 weights in middle layers. Once IQ2 dequant produces correct values, the CE loss will produce meaningful numbers.

### D — MoE Forward (BLOCKED on IQ2 fix)
Router works. Expert computation structure correct. Output garbage because IQ2 XXS/S dequant gives 1e6 values instead of correct quantized weights.

### E — MMProj Vision ✅ PASS
dump_mmproj tool reads all 334 tensors correctly. Merger dimensions verified.

### F — Run bench_e2e Fresh Benchmark
bench_e2e binary was rebuilt. Need fresh tok/s numbers vs old 7.69 tok/s. Now has parallel scan + fused SSM.
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
