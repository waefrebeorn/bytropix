# 7. Future Roadmap (May 15 PM v6 — Comprehensive)

## Integration Sprint — COMPLETE ✅
All 7 modules wired into integrated `train_integrated` binary. Flags verified individually + combined.

| Module | Status | Detail |
|--------|--------|--------|
| TST (bag+MCE) | ✅ | Bag s=8, MCE loss, 8/8 tests |
| RSGD Optimizer | ✅ | Riemannian SGD, valid ball constraint |
| Poincaré GQA | ✅ | Hyperbolic distance attention, forward+backward |
| Nested SSM K=4 | ✅ | 4 Poincaré balls, K=1/2/3 all pass |
| Nested MoE | ✅ | Poincaré hierarchy router, 396/396 tests |
| Data pipeline | ✅ | 1.07M tokens tokenized |
| CUDA kernels | ✅ | SSM scan + MoE dispatch |

**Training: 177s→11s/step (16×). Zero NaN. All P0-P2 complete.**

---

## Phase 7: GPU Optimization

| Task | Priority | Detail | Status |
|------|----------|--------|--------|
| GPU MoE forward | **P0** | Upload active expert weights → cuBLAS matmuls on GPU. Eliminates per-layer sync. Est: 11s→3s | ❌ Pending |
| GPU output projection | P1 | cublasSgemm for V=248320, replaces 2B CPU FMAs | ✅ **Done** |
| Double-buffer layers | P1 | Overlap GPU layer l+1 with CPU MoE layer l | ❌ Pending |
| Async D→H copies | P1 | Skip dead PGA copies when !pga_enabled | ✅ **Done** |

---

## Phase 8: Validation & Correctness

| Task | Priority | Detail | Status |
|------|----------|--------|--------|
| PGA LR tuning | **P1** | lr_gqa=lr*0.01→lr*0.001 or clip gradients at norm=1.0 | ❌ Pending |
| Multi-step convergence | P1 | 50+ step run, verify CE < 5.0 trend, no NaN | ❌ Pending |
| Expert utilization | P2 | Track load balancing entropy across 256 experts | ❌ Pending |

### Architecture Correctness (From Qwen3.6 Paper Audit)

Cross-reference C implementation against config.json findings:

| Parameter | Config | C Code | Action |
|-----------|--------|--------|--------|
| Q heads | 16 | → check | Verify |
| KV heads | **2** | → check | Verify (8:1 ratio) |
| Full attn head_dim | **256** | → check GQA_HEAD_DIM | Verify |
| Linear attn head_dim | **128** | → check SSM_D_STATE | Verify |
| Conv kernel dim | **4** | → check CONV_KERNEL | Verify |
| Partial RoPE factor | **0.25** (64/256 dims) | → check | Verify |
| RoPE theta | **10,000,000** | → check code constant | Verify |
| MRoPE 3D | section=[11,11,10] | ❌ Likely missing | **P2 implement** |
| bos/eos IDs | both **248044** | → check tokenizer | Verify |
| MTP head | 1 layer, shared embd | ❌ Missing | P3 implement |

---

## Phase 9: Vault Porting

### Vault Audit Results (12 Vaults Analyzed May 15)

| Vault | Code Exists | C Status | Impact | Action |
|-------|------------|----------|--------|--------|
| **c-training** | C (20 files, 70KB) | ✅ Running | Reference | Study for port patterns |
| **hamilton** | CUDA (llama.cpp fork) | ✅ External | Geodesic compression | Copy into bytropix/src/ |
| **hash-mind** | JAX V1-V7.1 | ✅ C port exists | Architecture ideas | Reference only |
| **attention** | PyTorch | ❌ None | O(n·k) linear | **Port sparse attention P2** |
| **optimizers** | JAX | ❌ None | Reusable LR logic | **Port Q-Controller + PID P2** |
| **theory** | Markdown | ❌ N/A | Math reference | No action |
| **lean-proofs** | Lean 4 | ❌ Incomplete | Verification | Low priority |
| **encoders** | Python | ❌ None | Research | Low priority |
| **phase3** | Python (66K lines) | ❌ None | Text-to-image | Low priority |
| **diffusion** | Python | ❌ None | Hyperbolic UNet | Research |
| **audio** | Python | ❌ None | Synthesis | Standalone |
| **draftPY** | Python (40+ scripts) | ❌ None | Ideas | Zero priority |

### Highest-ROI Port Targets

#### P2 — Sparse Attention (WuBu variant)
**Source:** `.hermes/vault/attention/` — clean PyTorch prototype
**Why:** O(n·k) linear complexity instead of O(n²). Perfect for long-context (262K native). Replaces GQA for context > 32K.
**Effort:** Medium — needs CUDA kernel, router for sparse/spread selection
**Status:** ❌ Not started

#### P2 — Q-Controller Meta-Optimizer
**Source:** `.hermes/vault/optimizers/` — JAX, 10-state×5-action Q-table
**Why:** Tiny, clean, high ROI. Learns optimal LR in training loop. Replaces hand-tuned schedule.
**Effort:** Low — ~200 lines C
**Status:** ❌ Not started

#### P3 — PID Lambda Controller
**Source:** `.hermes/vault/optimizers/` — JAX prototype
**Why:** Adaptive learning rate via PID control on loss gradient. Complements Q-Controller.
**Effort:** Low — ~150 lines C
**Status:** ❌ Not started

#### P3 — Hamilton Encoder Integration
**Source:** `~/HASHMIND/llama-cpp-rotorquant/` — existing CUDA kernels
**Why:** Geodesic compression for KV cache. ~3% overhead, ~62% memory reduction.
**Effort:** Medium — copy kernels, add tests
**Status:** ❌ Not integrated into bytropix src/

---

## Phase 10: Tailslayer Speculative Decode (NEW May 15)

**Source:** `~/HASHMIND/tailslayer/` (LaurieWired/tailslayer, Apache 2.0)
**Doc:** `THEORY/papers/tailslayer-*.md`, `.hermes/vault/tailslayer/`

### Pattern Match

| Tailslayer Concept | WuBuText Analog | Action |
|--------------------|----------------|--------|
| N replicas on independent DRAM channels | N draft tokens speculated in parallel | Port hedged-read pattern to CUDA kernel |
| clflush+reload timing | Forward pass timing | Adapt for spec-decode draft verification |
| Hedged read (first-response-wins) | Accept longest valid prefix | Implement `spec_verify_kernel` |
| Channel scrambling offset | Draft model distribution alignment | Wire into speculative decoding gate |
| N replicas pinned to separate cores | E experts dispatched across S SMs | MoE expert SM dispatch pattern |
| Physical address→channel mapping | CUDA shared memory bank conflict analysis | Add bank-conflict-aware shared mem layout |
| tREFI probe (clflush+TSC) | CUDA kernel launch overhead profiling | Add PCIe timing instrumentation |
| Sliding window pair sampling | Draft-target logit time alignment | Pair samples by timestamp, take min |
| Harmonic binning (periodicity detection) | Memory access pattern profiling | Periodicity analysis for cache optimization |

### Direct Port Plan (P2)
1. **Speculative Decode Kernel** (`spec_verify.cu`): Launch N draft verification threads across GPU SMs, first valid prefix wins, cancel remaining
2. **tREFI Probe for CUDA** (`pcie_probe.cu`): Port clflush+reload → CUDA event timing for PCIe transfer detection
3. **Sliding Window Pair Sampling**: Align draft-target logits by timestamp, take minimum latency
4. **Bank Conflict Analysis** (`bank_analyzer.cu`): Port `compute_channel()` → `compute_bank()` for shared memory

---

## Phase 11: Multi-Token Prediction (MTP)

Qwen3.6 has `mtp_num_hidden_layers=1` with shared embeddings.
- Adds 1 prediction head for next-2-token joint prediction
- No dedicated embeddings (`mtp_use_dedicated_embeddings=false`)
- Standard in Qwen3.5+ — missing from our C implementation
- **Impact:** Sampling efficiency improvement, but not blocking training
- **Priority:** P3

## Phase 11: Vision Encoder Verification

Cross-reference 27-layer vision transformer against Qwen3.6 config:
- `hidden_size=1152`, `intermediate_size=4304`, `num_attention_heads=16`
- `out_hidden_size=2048` → must match text model hidden_size
- `patch_size=16`, `temporal_patch_size=2`, `spatial_merge_size=2`
- **Verify:** Our vision implementation matches these numbers
- **Priority:** P3

---

## Phase 12: New Diagram Verification

| Diagram | File | Status | Description |
|---------|------|--------|-------------|
| Training Pipeline | `DIAGRAMS/training-pipeline.svg` | **NEW** | 6-stage training flow, metrics sidebar |
| Tailslayer Pattern | `DIAGRAMS/tailslayer-pattern.svg` | **NEW** | 8-row hedged-read → spec-decode analogy |
| Paper Audit | `DIAGRAMS/paper-audit.svg` | **NEW** | 14 Qwen3.6 params vs C implementation |

---

## Completed

- ✅ P0: Per-block IQ2_XXS extraction (177s→11s/step, raw_size fix 72→66)
- ✅ P1: Multi-flag verification (6 flags, all combos, 0 NaN)
- ✅ P2: MoE output magnitude (hidden max=13, was 5e9)
- ✅ P2: Memory optimization (persistent buffers in lmoe_t)
- ✅ GPU output projection (cublasSgemm for V=248320)
- ✅ Async D→H copies (dead PGA copies skipped)
- ✅ All 7 cold gaps closed
- ✅ NaN root cause fixed
- ✅ GPU vision pipeline (99ms, 0 NaN)
- ✅ Vault audit (13 vaults + tailslayer, ROI-ranked)
- ✅ Paper audit (32 Qwen files, arch discrepancy table)
- ✅ Tailslayer analysis (5 files, 8-pattern analogy map)
- ✅ 3 new SVGs (training-pipeline, tailslayer-pattern, paper-audit)
