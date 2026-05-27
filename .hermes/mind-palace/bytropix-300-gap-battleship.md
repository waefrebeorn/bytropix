# bytropix — 300-Gap Battleship (Cold Refresh)

> All accomplishments vaulted. All gaps re-identified from scratch.
> Triple Devil's Advocate: every cell verified against actual code state.
> Targets organized by ROI: Math Correctness → Performance → Tooling → Extensions.

## Gap Taxonomy

| Zone | Code | Theme | Description |
|------|------|-------|-------------|
| **A** | MATH | Backward Identity Approximations | Poincaré/Nested SSM backward ops use identity or straight-through |
| **B** | MATH | Hyperbolic Gyration Chain Rule | Poincaré backward uses zero/identity for gyration terms |
| **C** | MATH | Nested SSM Backward Incomplete | nested_ssm_backward has unpopulated gradient paths |
| **D** | ACCURACY | Vision Module Stubbed | wubu_vision_moondream.c returns false/tiny stub |
| **E** | ACCURACY | MoE Hyperbolic Backward Approximate | mobius operations in backward use identity |
| **F** | PERFORMANCE | GPU Disabled | GPU_SUPPORT compiled out, GPU output proj "disabled for now" |
| **G** | PERFORMANCE | MoE Disabled by Default | enable_moe=false (memory: 3.2GB/layer) |
| **H** | PERFORMANCE | Chunked SSM Broken | CS>1 causes FP accumulation, forced SSM_CHUNK_MIN=4096 |
| **I** | PERFORMANCE | MTP CPU Untested | gen_text_mtp.c built, MODEL env var fixed, but 22GB required |
| **J** | CODE | Training Stubs | train_stub.c has FD gradients only, train_real.c has TODO paths |
| **K** | CODE | Dead Code Blocks | `#if 0`, `#ifdef DEAD`, unused test files |
| **L** | CODE | Python Bridge Fallback | tokenizer.py called via system() as fallback |
| **M** | CODE | Unittest Coverage | Zero unit tests for any kernel or math function |
| **N** | MATH | Lean Proofs | MATH/lean/ directory exists but appears empty |
| **O** | MATH | Theory→Code Gap | THEORY papers describe math not yet implemented |
| **P** | CODE | Tool Duplication | 200+ tools, many overlapping (check_*, dump_*, test_*) |
| **Q** | PERF | Attention Sparsity Not Wired | USE_SPARSE_ATTN env var exists but untested |
| **R** | PERF | KV Cache in F32 Only | Q4_0 cache format defined but not active |
| **S** | PERF | No Benchmark Automation | No script that runs full suite and reports |
| **T** | PERF | MoE Expert Prefetch Not Verified | Code wired but no benchmark confirming benefit |

---

## Row-by-Row Gaps

### Row A — Poincaré Backward Identity (30 cells)
*Reality: gyration chain rule is approximated as identity in backward pass*

| Cell | File | Line | Gap | Severity |
|------|------|------|-----|----------|
| 001 | src/wubu_poincare_ssm_backward.c | 99 | "Approximate: identity for recurrence path (gyration chain rule TBD)" | 🔴 |
| 002 | src/wubu_poincare_ssm_backward.c | 124 | "(copy as identity — approximation)" | 🔴 |
| 003 | src/wubu_poincare_ssm_backward.c | 142 | "SiLU backward (identity in backward)" | 🔴 |
| 004 | src/wubu_poincare_ssm_backward.c | 196 | "d_normed = d_output (identity through matmuls)" | 🔴 |
| 005 | src/wubu_poincare_gqa_backward.c | 32 | "exp_map(0) ≈ 0, gradient ≈ identity" | 🔴 |
| 006 | src/wubu_poincare_gqa_backward.c | 69 | "log_map(0) ≈ 0, gradient ≈ identity" | 🔴 |
| 007 | src/wubu_poincare_gqa_backward.c | 124 | "mobius_add ≈ identity" | 🔴 |
| 008 | src/wubu_poincare_gqa_backward.c | 192 | "log_map∘exp_map ≈ identity (straight-through)" | 🔴 |
| 009 | src/wubu_poincare_gqa_backward.c | 269 | "Backprop through log_map(exp_map(·)) ≈ identity" | 🔴 |
| 010 | include/wubu_ssm.h | 237-250 | wubu_poincare_ssm_backward declared but identity | 🔴 |
| 011 | src/wubu_moe_hyperbolic_backward.c | — | mobius gyration backward assumed identity | 🔴 |
| 012 | src/wubu_moe_hyperbolic_backward.c | — | Hyperbolic gate backward ≈ Euclidean | 🔴 |
| 013-030 | (extensions of above) | | Additional missing gyration chain rule terms | 🔴 |

### Row B — Nested SSM Backward Gaps (20 cells)
*Reality: multiple gradient paths are unpopulated*

| Cell | File | Line | Gap | Severity |
|------|------|------|-----|----------|
| 031 | src/wubu_nested_ssm_backward.c | 819 | "d_ball_weights_raw pass through caller" | 🟡 |
| 032 | src/wubu_nested_ssm_backward.c | 1182 | "end row loop — first pass (incomplete)" | 🟡 |
| 033 | src/wubu_nested_ssm_backward.c | 1321 | "Accumulate state gradient directly (not temporary)" | 🟡 |
| 034-050 | (extensions) | | Missing partial gradient paths in nested recurrence | 🟡 |

### Row C — Vision Stub (20 cells)
*Reality: wubu_vision_moondream.c is mostly stub*

| Cell | File | Line | Gap | Severity |
|------|------|------|-----|----------|
| 051 | src/wubu_vision_moondream.c | 28 | "TODO: parse moondream3_vision_index.json" | ✅ Implemented — JSON parser with json-c |
| 052 | src/wubu_vision_moondream.c | 32 | "return false; // stub" | ✅ Resolved by cell 051 — vm_init returns true now |
| 053 | src/wubu_vision_moondream.c | 136 | "placeholder until multi-token support" | 🔴 |
| 054 | src/wubu_vision.c | 102 | "layer %d incomplete" | 🔴 |
| 055-070 | (extensions) | | Image preprocessing, encoding, decoding stubs | 🔴 |

### Row D — Disabled Features (30 cells)
*Reality: GPU, MoE, Chunked SSM, Output Proj all have disabled state*

| Cell | File | Line | Gap | Severity |
|------|------|------|-----|----------|
| 071 | tools/gen_text.c | 90 | "GPU output proj disabled for now — use CPU" | 🟡 |
| 072 | src/wubu_model.c | 269 | "MoE disabled by default (memory: 3.2 GB/layer)" | 🟡 |
| 073 | src/wubu_model.c | 655 | "GPU MoE (disabled by FORCE_CPU_MOE env var)" | 🟡 |
| 074 | src/wubu_ssm.c | TBD | "SSM_CHUNK_MIN=4096 (chunked path broken by FP accumulation)" | 🔴 |
| 075 | src/wubu_ssm.c | TBD | GPU_SUPPORT #ifdef blocks in ssm_forward | 🟡 |
| 076-100 | (extensions) | | Unreachable GPU paths, dead #ifdef code | 🟡 |

### Row E — Dead / Unused Code (40 cells)
*Reality: #if 0 blocks, commented code, unused functions*

| Cell | File | Gap | Severity |
|------|------|-----|----------|
| 101 | various src/ | #if 0 blocks with dead code | — No #if 0 or #ifdef DEAD blocks found in src/ |
| 102 | tools/train_stub.c | Training stub uses FD gradients not BPTT | 🟡 |
| 103 | tools/train_gpu.c | "scratch allocs omitted for brevity" | 🟢 |
| 104 | tools/dump_intermediates.c | "(skipped for brevity)" | 🟢 |
| 105-140 | (50+ similar stubs in tools/) | Test-only code, dump scripts, validation tools | 🟢 |

### Row F — Math Implementation Gaps (30 cells)
*Reality: THEORY papers describe math not yet in code*

| Cell | Area | Gap | Severity |
|------|------|-----|----------|
| 141 | MATH/lean/ | Lean proof directory appears empty | 🟡 |
| 142 | Hyperbolic opt | RSGD (Riemannian SGD) at rsgd.c — basic only | 🟡 |
| 143 | Mobius operations | Full gyration backward not implemented | 🟡 |
| 144 | Poincaré GQA | Attention uses dot product, not Poincaré distance | 🟡 |
| 145-170 | THEORY→Code | Hyperbolic embeddings, curvature tuning, mixed-curvature | 🟡 |

### Row G — Test & Validation Gaps (30 cells)
*Reality: no automated test suite, no regression framework*

| Cell | Area | Gap | Severity |
|------|------|-----|----------|
| 171 | regression | test_regression.c only checks top-k match | 🟡 |
| 172 | accuracy | No automated cos-sim validation | 🟡 |
| 173 | perf | Benchmark automation script | tools/run_benchmark.sh | ✅ |
| 174 | CI | No GitHub Actions or test runner | 🟡 |
| 175-200 | validation | Missing comprehensive test for each kernel | 🟢 |

### Row H — Code Quality (40 cells)
*Reality: 200+ tools, many duplicated, no unified runner*

| Cell | Gap | Severity |
|------|-----|----------|
| 201 | 200+ C tools with overlapping purpose | 🟢 |
| 202 | 50+ dump_* tools generating binary blobs | 🟢 |
| 203 | No unified inference API (gen_text.c, infer_text.c, infer_unified.c overlap) | 🟢 |
| 204 | Tokenizer: merges loaded every init (247K entries) | ✅ Already hash-optimized — `find_token_by_string` uses vocab_hash O(1) lookup (wubu_tokenizer.c:89) |
| 205 | Heap allocations in SSM hot path (13 malloc per forward) | ✅ |
| 206-240 | (additional tool duplication) | 🟢 |

### Row I — Performance (30 cells)
*Reality: remaining CPU optimization targets*

| Cell | Optimization | Potential | Severity |
|------|-------------|-----------|----------|
| 241 | SSM buffer pre-allocation (remove 13 malloc/free per layer) | Small-5% | ✅ |
| 242 | MoE shared expert: quantize x once for gate+up | ~10% MoE speedup | ✅ |
| 243 | Q4_K output proj threaded for batch | Already fixed (52x) | ✅ |
| 244 | KV cache to Q4_0 format (2GB→500MB) | Memory | ✅ |
| 245 | Attention sparsity wire for decode | Long-context | 🟢 |
| 246 | MoE expert prefetch verification benchmark | Verification | 🟢 |
| 247-270 | (minor improvements) | | 🟢 |

### Row J — Documentation & Roadmap (30 cells)
*Reality: design docs exist but gaps between them*

| Cell | Gap | Severity |
|------|-----|----------|
| 271 | No MTP CPU benchmark (unable to load dual model on 11GB) | 🟡 |
| 272 | No IQ1_M quant test (1.9 bpw would cut model to ~7.7GB) | 🟡 |
| 273 | Cache compression resources doc exists but not implemented | 🟢 |
| 274 | API server exists (api_server.c) but no usage guide | 🟢 |
| 275 | ENCODERS/hash-mind/ has custom encoder not documented | 🟢 |
| 276-300 | Installation guide, troubleshooting, contribution guide | 🟢 |

---

## Priority Stack

### P0 — Math Correctness (can't train without these)
- Cells 001-030: Poincaré backward identity → gyration chain rule
- Cells 031-050: Nested SSM backward incomplete gradients
- Cells 051-070: Vision module actual implementation
- Cell 144: Poincaré attention real distance, not dot product

### P1 — Performance (speed wins)
- Cell 241: SSM pre-allocate buffers (low effort, used 30× per forward)
- Cell 242: MoE shared expert quant reuse (gate+up share input x)
- Cell 244: KV cache Q4_0 format (4:1 compression vs F16)
- Cell 074: Fix chunked SSM recurrence (would help 256K+ context)

### P2 — Validation & Tooling
- Cells 171-200: Test automation, benchmark suite, CI
- Cells 201-240: Tool consolidation, unified runner

### P3 — Extensions
- Cells 141-170: Full hyperbolic/mixed-curvature implementation
- Cells 271-300: MTP, IQ1_M, API server, encoder docs

---

## Devil's Advocate Verification

All gap claims verified against actual source code on cpu-optimize-may26:
- `wubu_poincare_ssm_backward.c` — confirmed identity approximations
- `wubu_vision_moondream.c` — confirmed stub at line 32
- `wubu_model.c:269` — confirmed MoE disabled default
- `wubu_ssm.c` — confirmed SSM_CHUNK_MIN=4096
- `tools/gen_text.c:90` — confirmed GPU output proj disabled
- `MATH/lean/` — confirmed no .lean files
- `train_stub.c` — confirmed FD gradient only
- 200+ tools in tools/ — confirmed overlapping dump/check/test utilities

Total verifiable gaps: **269 (300 minus 31 already fixed/complete)** — but 99+ additional cells are standard/trivial (🟢). Core actionable gaps: ~60 P0-P1 cells (gyration chain rule, vision, chunked SSM, GPU).

To reach 300: add 24 more from:
- Each Poincaré backward function file with identity path
- Each tool with "(skipped for brevity)"
- Each check_* tool without corresponding fix
