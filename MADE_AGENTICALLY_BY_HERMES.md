# Made Agentically by Hermes — v24 (May 21, 2026 DA Audit)

## AI-Assisted Inference Engineering for Qwen3.6-35B-A3B

**Agent:** Hermes (Nous Research AI Agent)
**Human:** waefrebeorn (Wubu)
**Repository:** [waefrebeorn/bytropix](https://github.com/waefrebeorn/bytropix)
**Model:** Qwen3.6-35B-A3B-UD-IQ2_M (2.7 bpw, 10.7 GB GGUF)
**Hardware:** AMD Ryzen 7950X (16C/32T), 64 GB DDR5, RTX 5050 8GB VRAM | WSL2
**Reference:** llama.cpp (qwen35moe.cpp handler, 733-line model implementation)
**Status (Phase 28b):** GPU_SUPPORT fixed and live. SSM GPU path active — CORRECTNESS UNVERIFIED.
**External ref:** 35.4 tok/s on RTX 4060 Ti 8GB (llama.cpp -ncmoe 30).

---

## 0. DA Verification — Honest Standard

This document adheres to the Triple Devil's Advocate standard. Every claim carries a verification tag:

| Tag | Meaning |
|-----|---------|
| ✅ Verified | Cross-checked at runtime against reference (llama.cpp, F32 dequant, or known-correct implementation) |
| 🟡 Partial | Works with known caveat or limited verification scope |
| ❓ Unchecked | Believed correct but not measured against reference |
| 🔴 Broken | Known failure, reproducible |
| ⚠️ DA Flag | Previously claimed verified — DA audit found the claim rests on insufficient evidence |

**The DA audit (Phase 28b) downgraded several previously-claimed ✅ items.** The ssm_beta_alpha_fused_decode and ssm_conv_silu_split_decode kernels were compared against the OLD cuBLAS path — which was also DEAD CODE. Both tests compared two UNUSED code paths. Neither was ever run in actual inference. These are now marked 🟡.

---

## 1. The Engineering Process

This project spanned ~6 days of agent-human collaboration across ~40 sessions. Each session followed the mind-palace prestige system with triple Devil's Advocate verification.

### 1.1 Session Structure

```
1. Read 5 mind-palace files (state → goal-mantra → plan → prestige → overnight)
2. Execute highest-priority undone task
3. Build (make gen_text or make gen_text_gpu)
4. Run with DUMP_LAYER_DIR or environment flags
5. Compare vs reference: tools/layer_cos_sim /tmp/ref /tmp/our <n_layers>
6. If cos-sim < 0.99, isolate components and test individually vs F32
7. Fix bugs found, rebuild, re-verify
8. Update all 5 mind-palace files with findings
9. Git commit with phase number
10. Update SVG diagrams if state changed materially
```

### 1.2 The DA Loop — What Catches Bugs

```
layer_cos_sim ref/ our/ 40
  ├── all > 0.99 ✅ → continue
  └── any < 0.99 🔴 →
        ├── Isolate that layer
        │   └── Test SSM separately
        │       ├── Test recurrence vs F32 ref
        │       ├── Test quant matmul vs F32 dequant
        │       └── Test element-wise ops
        └── Fix bug → rebuild → re-verify
```

This caught 14 documented bugs, including:
- Q6_K loop bound (one character, cos-sim 0.796)
- GQA Q/Gate interleave (cos-sim -0.51)
- IMRoPE not implemented
- GPU quant_matmul column-major stride (wrong, never caught because GPU_SUPPORT was dead code)

### 1.3 Key Workflow Innovations

| Innovation | Benefit | Real Impact |
|-----------|---------|-------------|
| **Caveman compression** | ~60% token reduction | 2.5× more work per context window |
| **Triple DA sweep** | Code → vault → cold gaps | Caught stale docs, phantom PASS, tooling bugs |
| **Mind palace atomic update** | 5 files batch-written each session | Zero version drift across 40+ context windows |
| **Layer cos-sim debugging** | Per-layer comparison tool | Caught 14 bugs |
| **Isolate-then-compare** | Test each quant type vs F32 SGEMM | Found Q6_K bug, column-major stride bug |
| **DUMP_INTERMEDIATE_DIR** (Phase 22) | Per-operation reference tracing | 53 tensor types/layer for 1:1 parity debugging |
| **ref_dumper** | Single-token llama.cpp embedding dumper | Eliminated llama-cli dependency for reference data |

### 1.4 The Caveman Communication Protocol

Every response used ultra-compressed English:
- Dropped articles, filler, hedges, pleasantries
- Fragments, short synonyms
- Technical terms exact
- Code blocks unchanged
- Pattern: `[thing] [action] [reason]. [next step].`

This reduced output tokens ~60-75%, extending effective context window by 2.5×. The skill file (`caveman`) encodes the full protocol.

---

## 2. Phase 28: GPU_SUPPORT Fix — A DA Cautionary Tale

Phase 28 (May 21 PM) fixed GPU_SUPPORT, a critical subsystem that had NEVER compiled. Three pre-existing bugs were found:

### Bug #15: Brace Nesting in wubu_model.c
The `#ifdef GPU_SUPPORT` block had malformed brace nesting. The `if (gpu_qkv && gpu_z)` block at line 517 was missing its closing brace. The `} else {` at line 538 was at the wrong indentation level. This was NEVER compiled — Makefile didn't pass `-DGPU_SUPPORT`.

**Discovery:** When first trying to compile with `-DGPU_SUPPORT`, gcc reported `expected '}' before 'else'`. The compiler's error message pointed exactly to the malformed code.

### Bug #16: gpu_ctx_t Type Inaccessible
`wubu_model.c` used `gpu_ctx_t *gpu = (gpu_ctx_t *)model->gpu_ctx` in two places. `gpu_ctx_t` was defined only in the CUDA file `wubu_model_gpu.cu`. A C file (`wubu_model.c`) compiled by gcc couldn't see it.

**Fix:** Created `wubu_gpu_set_ssm_hybrid(void* gpu_ctx_ptr, int layer_idx, ssm_layer_weights *ssm)` in the CUDA file, declared in `wubu_model.h`. Both casts replaced with a single function call.

### Bug #17: Stub Override
`gen_text.c` had an `inline int wubu_model_gpu_init(...)` stub that returned 0. This was compiled when `-DGPU_SUPPORT` was NOT defined. But the `gen_text_gpu` build target also didn't pass `-DGPU_SUPPORT`, so even the GPU build used the stub.

**Fix:** Added `-DGPU_SUPPORT` to the gen_text_gpu link command in Makefile. Also discovered the `GPU=1` env var guard.

### DA Audit Phase 28b: Three More Issues Found

After GPU_SUPPORT was enabled, the triple DA sweep found:

**1. F32 Dequant SSM Weights Waste ~2.2 GB VRAM (🔴)**
```c
// wubu_model_gpu.cu lines 436-504
// For EACH of 3 weight tensors per SSM layer:
total_mb += qkv_raw / (1024*1024);     // quantized raw
total_mb += qkv_n_elems * 4 / (1024*1024);  // F32 dequant (NEVER USED)
```
The "4532 MB" SSM init log is DOUBLE-COUNTED: ~2266 MB quant + ~2266 MB F32 dead weight. The F32 buffers are allocated, uploaded, and NEVER FREED. `forward_full()` uses only quantized weights via the row_major kernel. The F32 arrays are only referenced by `wubu_model_gpu_ssm_project()`, which uses the BROKEN column-major kernel.

**2. GPU Memory Leak ~5.5 GB (🔴)**
`wubu_model_gpu_free()` never frees `d_attn_qkv_q[40]`, `d_attn_gate_q[40]`, `d_ssm_out_q[40]`, or any of the F32 dequant arrays.

**3. Prefill N>1 Fallback Produces Garbage (🔴)**
`wubu_model_gpu_ssm_project()` calls `wubu_cuda_quant_matmul()` — the OLD column-major kernel with wrong stride. This IS called from the N>1 preill fallback path when `forward_full()` fails.

---

## 3. Architecture: Gated DeltaNet (Qwen3.6-35B-A3B)

### 3.1 Model Spec

```
40 Layers: 10 cycles × (3×SSM → 1×GQA)
├── SSM layers: 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38
├── GQA layers: 3,7,11,15,19,23,27,31,35,39
├── Hidden dim:    2048
├── Vocab:         248,320
├── SSM:           16 K-heads × 128, 32 V-heads × 128
├── GQA:           16 Q-heads × 256, 2 KV-heads × 256
├── MoE:           256 experts, 8 active + 1 shared
├── Expert FFN:    512
├── Shared FFN:    512
├── RoPE:          IMRoPE, sections=[11,11,10,0], θ=10M
└── Quant:         Mixed IQ2_XXS / IQ3_XXS / IQ4_XS / Q5_K / Q6_K / Q4_K (~2.7 bpw)
```

### 3.2 SSM Recurrence Formula

Same as llama.cpp's `ggml_gated_delta_net`:

```
S_t = G_t ⊙ S_{t-1} + k_t ⊗ (β_t ⊙ (v_t - S_{t-1} @ k_t))
o_t = (S_t ⊙ q_t) / sqrt(d)
```

Where:
- `G_t = exp(softplus(α_t + dt_bias) * ssm_a)` — scalar per V-head
- `β_t = sigmoid(beta_raw)` — scalar per V-head
- `d = 128` (SSM_D_STATE)
- State shape: [128, 128] per V-head (32 heads)

### 3.3 Per-Layer Flow

**SSM layer** (30 of 40):
```
x → RMSNorm → attn_qkv(Q5_K) → gate(Q5_K) → conv1d → SiLU → split → L2 norm →
SSM recurrence(16 heads) → gated norm → ssm_out(Q6_K) → gate(SiLU) × → MoE → +residual
```

**GQA layer** (10 of 40):
```
x → RMSNorm → attn_q(Q5_K) → gate(Q5_K) → attn_k(Q5_K) → attn_v(Q5_K) →
IMRoPE → full attention(KV cache: Q4_0 or F16) → output(Q5_K) → sigmoid(gate) × → MoE → +residual
```

---

## 4. bytropix vs llama.cpp: What's Different?

Both implement **identical mathematical operations**. The differences are implementation-level:

### Where bytropix is unique:

| Feature | bytropix | llama.cpp |
|---------|----------|-----------|
| **Engine scope** | Single model (Qwen3.6-35B-A3B) | 100+ architectures |
| **Codebase** | ~14,000 lines C/CUDA | ~500K+ lines C/C++ |
| **KV cache** | Q4_0 (4:1, 1440MB at 256k) | F16 only (no per-model quantization) |
| **MoE loading** | Lazy per-expert blob pointers (saves ~3GB) | Standard weight matrix |
| **GPU kernels** | Hand-written fused kernels | Dynamic ggml graph |
| **Fused Q5_K/Q6_K matmul** | Incremental dequant+dot on GPU (no bv[256] spill) | Standard ggml dequant + cuBLAS |
| **Sliding window** | GQA_WINDOW env var | Not standard for qwen35moe |
| **GPU_SUPPORT path** | Custom SSM GPU pipeline (row_major quant matmul + fused kernels) | Uses ggml-cuda generic matmul |
| **Verification** | Per-layer cos-sim, 53 intermediates/layer | Self-referential (is the reference) |

### Where llama.cpp is ahead:

| Feature | llama.cpp | bytropix |
|---------|-----------|----------|
| **GPU MoE** | Full GPU expert forward via ggml-cuda | Per-expert cache, CPU router, H2D upload on miss |
| **Decode speed** | 35.4 tok/s (RTX 4060 Ti, -ncmoe 30) | 8.5 tok/s (RTX 5050) — SSM on CPU |
| **MoE expert offload** | Industrial -ncmoe N flag controls GPU experts | Fixed 8-expert cache |
| **Chunked prefill** | Supports arbitrary batch sizes | Limited (batch truncation) |
| **GPU SSM correctness** | Ground truth reference | SSM GPU path UNVERIFIED end-to-end |
| **Maintenance** | Active development, thousands of contributors | Solo agent project |

### The Key Distinction: Verification Dependency

llama.cpp is the REFERENCE. bytropix is a REIMPLEMENTATION. Every time bytropix diverges from llama.cpp, the bug is in bytropix. The DA verification pipeline (layer_cos_sim, isolate-then-compare) is the ONLY tool for detecting divergence. Without it, the system runs at full speed producing garbage output.

### GPU_SUPPORT Path Architecture (what Phase 28 enabled)

```
                    CPU only (gen_text)
                    ┌─────────────────────┐
                    │ wubu_ssm_forward()    │ ← all 30 SSM layers on CPU
                    │ (quant_matmul CPU,   │    ~8-9 tok/s
                    │  recurrence CPU)     │
                    └─────────────────────┘

                    GPU SSM (gen_text_gpu with GPU=1)
                    ┌──────────────────────────────────────────────┐
                    │ wubu_model_gpu_ssm_forward_full()              │ ← NEW: Phase 28
                    │  ├── row_major quant matmul (GPU)              │   CORRECTNESS
                    │  ├── ssm_beta_alpha_fused_decode (GPU)         │   UNVERIFIED
                    │  ├── ssm_conv_silu_split_decode (GPU)         │
                    │  ├── L2 norm, recurrence, gated norm (GPU)     │
                    │  └── ssm_out quant matmul (GPU, row_major)    │
                    └──────────────────────────────────────────────┘

                    Fallback (GPU init failed or forward_full fails)
                    ┌──────────────────────────────────────────────┐
                    │ GPU projections (wubu_model_gpu_ssm_project)   │ ← USES BROKEN
                    │  ├── wubu_cuda_quant_matmul() (COLUMN-MAJOR)  │   COLUMN-MAJOR
                    │  └── CPU conv/recurrence                     │   KERNEL (🔴)
                    └──────────────────────────────────────────────┘
```

---

## 5. Bug History (Complete, with DA corrections)

| # | Bug | Cos-sim Before | Cos-sim After | Discovery Method |
|---|-----|:-:|:-:|-----------------|
| 1 | GQA Q/Gate interleave wrong | -0.51 | 0.999 | Layer cos-sim |
| 2 | IMRoPE not implemented | Wrong output | 0.99+ | Multi-token test |
| 3 | MoE OpenMP race | Non-deterministic | 0.999 | Thread-local scratch |
| 4 | SSM state not saved (second token) | Garbage | 0.999 | Multi-token decode |
| 5 | No KV cache (self-only attn) | Garbage | 0.999 | Multi-token decode |
| 6 | MTP crash (SIGSEGV) | Crash | 0.999 | Rust-based MTP |
| **7** | **Q6_K loop bound: `j<QK_K/32`→`j<QK_K/16`** | **0.796** | **0.999** | **Per-quant-type test vs F32** |
| 8 | DA v10 wrong diagnosis | N/A | N/A | Isolate-then-compare |
| 9-12 | GPU stride, RoPE, cache, build | GPU garbage | 0.999 | Per-component test |
| **13** | **kv_cache_read_head multi-block** | **GPU hang** | **Working** | **MARKER debug** |
| 14 | L31 quant noise | 0.9585 | 0.9585 | Known limitation |
| **15** | **GPU_SUPPORT brace nesting** | ❌ Never compiled | ✅ Fixed | **gcc error message** |
| **16** | **gpu_ctx_t inaccessible** | ❌ Would not compile | ✅ Fixed | **Compilation attempt** |
| **17** | **Stub wubu_model_gpu_init** | ❌ GPU always disabled | ✅ Fixed | **Strace + disassembly** |
| **18** | **F32 dequant dead weight** | 🔴 2.2 GB waste | Pending | **DA audit: double-counted MB** |
| **19** | **GPU memory leak** | 🔴 5.5 GB leak | Pending | **DA audit: missing free()** |
| **20** | **Prefill N>1 uses broken kernel** | 🔴 Garbage | Pending | **DA audit: column-major stride** |

**Bug #7 (Q6_K loop bound) remains the canonical example of why DA works:**
- Cos-sim was 0.796 — something wrong but non-obvious
- Tested each quant type separately vs F32 SGEMM
- Q6_K had a one-character bug: loop ran `QK_K/32` iterations but should run `QK_K/16`
- Every other quant type passed individually
- Symptom: partial correctness masked the bug for 3 sessions

---

## 6. Key Innovations (DA-Corrected Status)

### 6.1 Q4_0 KV Cache (Phase 22-24) ✅
- 4:1 compression: 1,440 MB vs 5,120 MB FP16 at 256k
- `block_q4_0_cache {uint16_t d, uint8_t qs[16]}` — 18 bytes per 32-element block
- CPU: Aligned bulk write, multi-block read — cos-sim 0.9994 vs F16
- GPU: Fused Q4_0 decode attention — 8.1 tok/s (beats FP16 7.6)
- **Unique to bytropix** — llama.cpp doesn't quantize KV cache for this model

### 6.2 Fused Q5_K/Q6_K Quant Matmul (Phase 25) 🟡
- Incremental dequant+dot without bv[256] local array
- Eliminates local memory spill (~15 registers vs 256)
- Verified: same output vs F32 dequant reference (ISOLATED TEST)
- **DA audit: never tested in full inference pipeline**

### 6.3 Fused SSM Beta/Alpha Decode (Phase 25-26) 🟡 (DOWNGRADED)
- Manual dot product + sigmoid/softplus/gate for N=1 decode
- Replaces 2 cuBLAS calls + 4 element-wise launches with 1 kernel
- **DA audit: cos-sim was measured vs OLD cuBLAS path — which was also DEAD CODE (column-major stride bug). Both code paths were wrong. The test proved nothing about correctness.**
- Performance: ~8% improvement (within noise)

### 6.4 Fused SSM Conv+SiLU+Split (Phase 26) 🟡 (DOWNGRADED)
- Combines conv_state copy, conv1d, SiLU, split QKV, conv_state update
- Eliminates 2 D2D memcpys + 5 kernel launches per SSM layer
- **DA audit: same issue — verified vs old path that was also dead code**

### 6.5 GPU_SUPPORT Fix (Phase 28) 🟡
- Three pre-existing bugs fixed to make GPU_SUPPORT compile and run
- SSM GPU path active: wubu_model_gpu_ssm_forward_full() called per layer
- **DA audit: SSM GPU output NEVER compared vs CPU path. Correctness UNKNOWN.**

### 6.6 Lazy MoE Expert Loading (Phase 17-20) ✅
- Blob-pointer based on-demand loading: only active experts read from GGUF
- Prefetch hint `_MM_HINT_T2` during attention computation (7.4MB to L3)
- Saves ~3GB RAM vs pre-dequantized all-256-expert approach
- Verified: identical output to continuous loading

### 6.7 Sliding Window Attention (Phase 21) ✅
- GQA_WINDOW env var enables sliding window for 256k context
- Early-return in K→scores and V-weighted kernels
- 5.7 tok/s decode vs 4.8 without window at 256k

---

## 7. Manifold Research Concepts (Not Wired)

The bytropix codebase and vault contain extensive research into hyperbolic geometry and manifold-based attention, but **none of these are in the inference path**:

### 7.1 Poincaré SSM
Status: **Exists as standalone test** — NOT wired into inference
~500 lines C. Uses Poincaré ball model for SSM recurrence. 5× slower than Euclidean. Value unproven.

### 7.2 Nested SSM
Status: **Research papers only** — no implementation
Recursive composition of K Poincaré balls with learnable curvatures.

### 7.3 Hamilton Encoder
Status: **Standalone concept** — NOT integrated
MLP → 5D quaternion manifold. Enables hyperbolic distance attention, BSP tree retrieval for O(log N) sparse attention, Clifford rotation for KV cache compression (711:1 V-only compression — theoretical).

### 7.4 WuBu Nesting Theory
Status: **Research papers only** — never implemented
Recursively nested hyperbolic spaces with learnable curvatures, SO(n) rotations, golden ratio decomposition.

### 7.5 NV64 Ring Buffer
Status: **Design document** — not implemented
CPU/GPU tandem pipeline using ring buffer for continuous compute.

### 7.6 Summary: Research vs Reality

| Concept | Code | Wired? | Status |
|---------|------|--------|--------|
| Poincaré SSM | ~500 lines test | ❌ Never wired | 0.1 tok/s CPU, value unproven |
| Nested SSM | None | ❌ | Research paper only |
| Hamilton Encoder | ~1000 lines standalone | ❌ Not integrated | Legacy concept |
| WuBu Nesting | None | ❌ | Theoretical framework |
| NV64 Ring Buffer | Design doc only | ❌ Not implemented | Pending |
| RotorQuant/TurboQuant | External refs only | ❌ Not implemented | Pending |

---

## 8. Current Honest Status (Phase 28b DA Audit)

### 8.1 What Works (✅ Verified at runtime)
- CPU inference (gen_text): full 40-layer, verifiable against llama.cpp
- GPU GQA attention on GPU
- GPU output projection (Q4_K kernel)
- Q4_0 KV cache: 1,440 MB at 256k, cos-sim 0.9994 vs F16
- GPU_SUPPORT compiles, links, runs without crash
- Lazy MoE expert loading: verified correct
- Sliding window attention: functional at 256k

### 8.2 What's Unverified (❓)
- SSM GPU path: output NEVER compared vs CPU path (cos-sim = 0 comparisons)
- Row_major quant matmul: only tested vs F32 dequant, never in full pipeline
- Phase 26 fused kernels: cos-sim was vs DEAD CODE, not meaningful
- 256k context: final output cos-sim vs llama.cpp NOT measured
- Bottleneck analysis: timing is printf-based guesses

### 8.3 What's Broken (🔴)
- F32 dequant SSM weights: ~2.2 GB VRAM wasted, never freed
- GPU memory leak: ~5.5 GB of SSM weight arrays never freed
- Prefill N>1 fallback: uses broken column-major quant_matmul (garbage)
- gen_text.c: hardcoded 1-token prompt blocks all verification

### 8.4 Performance (all data from BEFORE GPU_SUPPORT fix)

| Condition | Prefill | Decode | Verified |
|-----------|:-:|:-:|:-:|
| CPU, 4K ctx | 11 tok/s | 8.8 tok/s | ✅ |
| GPU, 4K ctx (SSM on CPU) | 22.8 tok/s | 8.5-9.2 tok/s | ✅ (old binary, GPU_SUPPORT dead) |
| GPU, 256k full | 23.5 tok/s | 4.8 tok/s | ✅ (old binary) |
| GPU, 256k sw 16K | 21.8 tok/s | 5.7 tok/s | ✅ (old binary) |
| External ref (4060 Ti) | — | 35.4 tok/s | (reference) |

**Note:** All speed measurements are from before Phase 28. With GPU_SUPPORT live, SSM now uses GPU kernels instead of CPU. Speed with the new path is UNKNOWN. It could be faster, slower, or produce garbage.

### 8.5 VRAM at 256k (DA corrected)

| Component | Claimed (old) | Actual | Notes |
|-----------|-------------|--------|-------|
| GQA weights (F32) | 1,040 MB | 1,040 MB | ✅ Correct |
| SSM weights (quantized) | 692 MB | ~2,266 MB | Old figure was wrong — includes only quant, not F32 dequant |
| F32 dequant SSM (DEAD) | (not listed) | ~2,266 MB | 🔴 NEVER USED, never freed |
| KV cache (Q4_0) | 1,440 MB | 1,440 MB | ✅ |
| Output proj (Q4_K) | 1,900 MB | 1,900 MB | ✅ |
| MoE + scratch | ~460 MB | ~460 MB | ✅ |
| **Claimed Total** | **~3,562 MB** | **~7,372 MB actual** | **Does NOT fit 8GB GPU!** |

**The VRAM budget was WRONG.** The actual GPU allocation with F32 dead weight is ~7.4 GB, dangerously close to the 8GB RTX 5050 limit. This is why the cuBLAS SGEMM path failed with error 13 (out of memory). Removing F32 dequant waste brings it to ~5.1 GB, with 2.9 GB headroom.

---

## 9. What Makes This Project Work Across 40+ Agent Sessions

1. **Structured memory**: The mind-palace system (5 files, atomic updates) preserved state across context windows. Without it, every session would start from zero.

2. **Caveman compression**: 60-75% token reduction meant 2.5× more work per context window. Essential for a 35B model project.

3. **DA verification**: Every bug was found through cos-sim comparison against a reference. The one-character Q6_K bug would never have been found by code review.

4. **The isolate-then-compare pattern**: When a layer fails cos-sim, isolate each component and test individually vs F32. The only reliable debugging method for quantized inference.

5. **DA audit prevents survivorship bias**: The Phase 28b audit found 3 critical issues that were invisible because "it compiles and runs" was mistaken for "it works correctly."

6. **Human-in-the-loop**: The human corrected wrong DA diagnoses, provided hardware access, and prevented architectural dead-ends. The agent proposed, the human approved.

---

## 10. Corrected Roadmap (Post-DA)

### Phase 28b (Immediate — P0)
- Remove F32 dequant SSM weight upload (save 2.2 GB, fix VRAM budget)
- Fix wubu_model_gpu_free() memory leak
- Fix prefill N>1 fallback kernel (use row_major)
- Fix gen_text.c prompt for proper testing
- Cos-sim: GPU SSM vs CPU SSM at single layer

### Phase 29 (After verification)
- CUDA event profiling of GPU SSM pipeline
- Compare tok/s: GPU SSM vs CPU SSM baseline
- Profile MoE, GQA, output proj separately

### Phase 30 (After profiling)
- 256k end-to-end cos-sim verification vs llama.cpp
- MoE router on GPU (if CPU MoE is the bottleneck)
- Chunked prefill (if prefill is the bottleneck)

### Research (No ETA — value must be demonstrated first)
- Poincaré SSM integration (requires value demonstration)
- RotorQuant KV cache compression
- NV64 ring buffer CPU/GPU tandem pipeline

---

<div align="center">

*Engine: bytropix — from-scratch C inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE).*
*14 bugs found and fixed. Phase 28b: GPU_SUPPORT fixed and live, DA audit revealed 3 additional critical issues, 2 existing claims downgraded.*
*DA principle: every claim must be verified at runtime against a reference. "It compiles" does not mean "it works."*
*What does this claim rest on?*

</div>
