# Overnight Map — Phase 28o: Triple DA Complete, All Markdown Cleaned

**Active repo:** /home/wubu/bytropix/
**Main model:** /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (11.5GB, 40 layers)
**MTP model:** /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf (11.9GB, +blk.40 head)
**Current commit:** 473f2b2 (pushed to origin/master)

## Session Summary (May 21, 2026 — Triple DA Run)
~200 tool calls, 7 markdown files rewritten, 2 Makefile fixes, 6 sm_120 bugs documented, 10 tools copied to vault, 1 new DA synthesis doc.

### What Was Done

**1. Triple Devil's Advocate Run**
- DA-1 (Code vs Claims): Read all 8 core mind-palace files + 3 vault papers. Found 7 stale files.
- DA-2 (Vault Deep Dive): Read DeepSeek-V3, DeepSeek-V3.2, DeepSeekMoE papers. Cross-referenced against plan.md. Found 6 P2 gaps (NSA, sigmoid gating, load balancing, FP8, chunked prefill, RoPE).
- DA-3 (Hardware): Audited RTX 5050 utilization. Found FP8 Tensor Cores, CUDA Graphs, async H2D, multi-block parallelism all unused.

**2. Markdown Files Updated (7 files)**
| File | What Changed |
|------|-------------|
| state.md | Sharpened P2 status, added sm_120 bug table, commit log |
| goal-mantra.md | Updated priorities to P2 hardware utilization |
| plan.md | P0-P1 closed, P2 reprioritized with vault references |
| prestige_prompt.md | Added DA v13 findings, P2 queue, sm_120 bugs |
| ARCHITECTURE.md | **REWRITE** — May 18→May 21: added GPU, MTP, vision, DA v13, sm_120 bugs, performance tables |
| testing.md | **REWRITE** — May 16 'broken claims' → accurate CPU works |
| project.md | **REWRITE** — May 16 'broken inference' → accurate state |

**3. Makefile Fixed** — `gen_text_cpu` target was broken (GPU symbols in CORE_OBJ). Added `src/wubu_moe_cpu.o` as CPU-only wubu_moe variant.

**4. CUDA sm_120 Bugs (6 documented):**
- Bugs 1-3: Original skill (static __shared__, syncthreads+reduce, extern uint8_t)
- Bug 4: compute-sanitizer WDDM unavailable
- Bug 5: WDDM memory/context init (2-5s)
- Bug 6: Divergent warp primitives
- Added to cuda-sm120-bugs skill

**5. Tools Copied to Vault (.hermes/vault/tmp-tools/):**
10 key debug + reference tools with README documenting all env vars

**6. Reference Data Pipeline Verified:**
```bash
# 40 hidden states + 1997 intermediate tensors per BOS token
DUMP_LAYER_DIR=/tmp/ref_layers ./ref_dumper model.gguf  # 40 files, 8KB each
DUMP_INTERMEDIATE_DIR=/tmp/ref_interm ./ref_dumper model.gguf  # 1997 files
```

## What's Blocked
- gen_text still won't build with GPU objects (need libcuda for wubu_model_gpu_*)
- gen_text_cpu works (verified: coherent text, single BOS token prefill verified)
- GPU text inference remains net-negative (CPU-only 2-5x faster)

## Verifiable Facts (DO NOT RE-DERIVE)
- ref_dumper + DUMP_LAYER_DIR = 40 files × 8192 bytes (1 BOS token)
- ref_dumper + DUMP_INTERMEDIATE_DIR = 1997 intermediate files (~9MB)
- gen_text_cpu builds with `make gen_text_cpu`
- layer_cos_sim builds with `make layer_cos_sim`
- cuda-sm120-bugs skill updated: 6 bugs, build flags, verified configs table
- sm_120 FP8 Tensor Cores available but unused
- All markdown files committed at 473f2b2

## Next Session: P2 Hardware Utilization
1. GPU RMSNorm + SiLU kernels (low impact, cleanup)
2. Chunked prefill (infrastructure exists in wubu_ssm_chunked.c)
3. RoPE extrapolation 4x (single param)
4. NSA sparse attention (high impact for 256K)
5. Sigmoid gating + load balancing (DeepSeekMoE)
6. FP8 Tensor Cores (sm_120 native, 2x throughput potential)
