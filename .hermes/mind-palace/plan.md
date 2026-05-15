# WuBuText AI — Plan (May 15 PM v6, Comprehensive)

## Purpose
P0-P2 complete. Training at 11s/step (16×), 0 NaN. This plan spans GPU optimization, model correctness (unimplemented Qwen features found in paper audit), and vault porting (12 vaults audited).

---

## Priority Queue

### P0 — GPU MoE Forward
Current MoE on CPU forces per-layer `cudaStreamSynchronize`. Moving to GPU eliminates it entirely.
- **Design:** Upload active expert weights (8 × 3 × 4MB = 96MB) to GPU buffer once per step
- **Call `wubu_cuda_moe_dispatch`** (already exists as CUDA kernel)
- **Impact:** Estimated 11s → ~3s/step (eliminate 40 sync points)
- **Risk:** 3GB full expert buffer doesn't fit VRAM → need sparse upload

### P1 — PGA LR Tuning
PGA backward gradients extreme (dQ=1.95, dK=0.004, dV=0.70, dX=571). Current lr_gqa=lr*0.01=1e-5 too high.
- **Fix:** Try lr_gqa = lr * 0.001 or gradient clipping at norm=1.0
- **Impact:** Steps 2+ would not jump from CE 21.6→69 (currently stuck)

### P1 — Multi-Step Convergence (50+ steps)
Current verification only at 2-3 steps. Need to verify:
- No long-term NaN emergence
- CE steadily decreasing (target < 5.0 after many steps)
- Embedding norms stable (no drift to Poincaré boundary)

### P2 — MRoPE (Multi-Resolution RoPE) Implementation
**Found in paper audit: only 0.25 partial rotary factor used, not MRoPE.**
- Qwen3.6 uses `mrope_interleaved=true`, `mrope_section=[11,11,10]` (32 total 3D positional dims)
- Our code may use standard 2D RoPE — needs code audit
- **Impact:** If wrong, position encoding degrades at long context (>32K)
- **Fix:** Verify RoPE implementation against config.json `rope_theta=10,000,000`, `partial_rotary_factor=0.25`

### P2 — Architecture Correctness Checks (From Paper Audit)
Cross-reference C implementation against Qwen3.6-35B-A3B config.json:
- **Head dims:** full attn=256, linear attn=128 (our code must use per-type head dim)
- **KV heads:** 2 (8:1 GQA ratio, not 4)
- **Conv kernel:** `linear_conv_kernel_dim=4` in linear attention path
- **Output gate:** sigmoid (default), swish for 27B variant
- **MoE shared expert:** present in Qwen3.5+ (our code has it ✅)
- **Token IDs:** bos=248044, eos=248044 (SAME token!), pad=null

### P2 — Vault Porting (Highest ROI Sorted)

| Vault | Code Exists | ROI | Effort | Action |
|-------|------------|-----|--------|--------|
| **Sparse Attention** | PyTorch prototype | High — O(n·k) linear | Medium | Port to CUDA kernel, replace GQA for long context |
| **Q-Controller Optimizer** | JAX, 10-state×5-action | High — reusable | Low (small, clean) | Port to C, add as training flag |
| **Tailslayer** | C++ hedged reader + tREFI probe | High — spec-decode pattern match | Medium | Port hedged-read CUDA kernel for draft verification |
| **Hamilton Encoder** | External CUDA in llama.cpp fork | Medium — geodesic compression | Medium | Copy CUDA kernels into bytropix/src/ |
| **PID Lambda Controller** | JAX prototype | Medium — adaptive LR | Low | Port to C, wire into training |
| **Toroidal Gradients** | JAX examples | Low — experimental | Low | Research only |
| **HGA-UNet Diffusion** | Python only | Low — compute heavy | High | Not yet — theory validation needed |
| **WuBuSynth Audio** | Python only | Low — tangential | High | Standalone project |

### P2 — Tailslayer-Inspired Architecture (May 15 Findings)

Tailslayer ([LaurieWired/tailslayer](https://github.com/LaurieWired/tailslayer), Apache 2.0) is a C++ library that reduces DRAM refresh tail latency via hedged reads across independent memory channels. Its patterns map directly to LLM inference and training:

| Tailslayer Concept | WuBuText Analog | Action |
|--------------------|----------------|--------|
| **N replicas on independent DRAM channels** | N draft tokens speculated in parallel | Port hedged-read pattern to CUDA kernel |
| **clflush+reload timing** | Forward pass timing | Adapt for spec-decode draft verification |
| **Hedged read (first-response-wins)** | Accept longest valid prefix | Implement `spec_verify_kernel` |
| **Channel scrambling offset** | Draft model distribution alignment | Wire into speculative decoding gate |
| **N replicas pinned to separate cores** | E experts dispatched across S SMs | MoE expert SM dispatch pattern |
| **Physical address→channel mapping** | CUDA shared memory bank conflict analysis | Add bank-conflict-aware shared mem layout |
| **tREFI probe (clflush+TSC)** | CUDA kernel launch overhead profiling | Add PCIe timing instrumentation |
| **Sliding window pair sampling** | Draft-target logit time alignment | Pair samples by timestamp, take min |
| **Harmonic binning (periodicity detection)** | Memory access pattern profiling | Periodicity analysis for cache optimization |

**Files examined:**
- `hedged_reader.hpp` — 221-line C++ template for N-way hedged reads across DRAM channels
- `trefi_probe.c` — 335-line DRAM refresh jitter detector (clflush+reload, TSC calibration, harmonic binning)
- `discovery/benchmark/main.cpp` — Multi-arm benchmark (single_quiet, hedged_quiet, single_stress, hedged_stress)
- `discovery/benchmark/benchmark.cpp` — Measurement thread with warmup, paired-sample analysis via sliding window
- `discovery/benchmark/hw_utils.hpp` — Core pinning, TSC calibration, virtual-to-physical address resolution

**Direct port opportunities:**
1. **Speculative Decode Kernel** (`spec_verify.cu`): Use hedged-read template pattern — launch N draft verification threads across GPU SMs, first valid prefix wins, cancel remaining
2. **Bank Conflict Analysis** (`bank_analyzer.cu`): Port `compute_channel()` → `compute_bank()` for CUDA shared memory
3. **tREFI Probe for CUDA** (`pcie_probe.cu`): Port clflush+reload pattern → CUDA event timing for PCIe transfer detection
4. **MoE SM Load Balancer** (`moe_sm_dispatch.cu`): Port N-replica→M-channel mapping → E-expert→S-SM mapping

### P3 — MTP Head Implementation
Qwen3.6 has `mtp_num_hidden_layers=1` (multi-token prediction).
- Predicts next 2 tokens jointly
- Uses shared embeddings (`mtp_use_dedicated_embeddings=false`)
- **Impact:** Better sampling efficiency at inference time
- **Effort:** Medium — needs separate output head + loss

### P3 — Vision Encoder Verification
Cross-reference 27-layer vision transformer against Qwen3.6 config:
- `hidden_size=1152`, `intermediate_size=4304`, 16 heads
- `out_hidden_size=2048` (matches text model)
- `patch_size=16`, `temporal_patch_size=2`, `spatial_merge_size=2`
- **Verify:** Our vision implementation matches these numbers

---

## Vault Summary (12 Vaults Audited May 15)

| Vault | Description | C Code Status | Recommended Action |
|-------|-------------|--------------|-------------------|
| **c-training** | Pure C transformer (hash-attention) | ✅ Running, 70KB, ~4000 steps/sec | Reference for porting patterns |
| **hamilton** | Geodesic encoder | ✅ External CUDA (llama.cpp fork) | Copy in, add test |
| **hash-mind** | WuBuMind JAX (V1-V7.1) | ✅ C port in c-training/ | Study for architectural ideas |
| **attention** | 4 variants (sparse, hyperbolic, topological, entropix) | ❌ Python/PyTorch only | Port sparse attention P2 |
| **optimizers** | Q-Controller, PID | ❌ JAX only | Port both P2 |
| **theory** | WuBu physics | ❌ No C needed | Reference for math |
| **lean-proofs** | Lean 4 formal proofs | ❌ 4 incomplete proofs | Low priority |
| **encoders** | Symmetric AE → QAE → generative | ❌ Python only | Research |
| **phase3** | Text-to-image pipeline | ❌ Python, 66K lines | Low priority |
| **diffusion** | HGA-UNet | ❌ Python only | Research |
| **audio** | WubuSynth | ❌ Python only | Standalone |
| **draftPY** | 40+ experimental scripts | ❌ Python only | Idea source |

## Research Paper Discrepancies (From Qwen3.6-35B-A3B Config.json)

| Parameter | Config Value | Our Code | Action |
|-----------|-------------|----------|--------|
| `num_attention_heads` | 16 | → check header | Verify |
| `num_key_value_heads` | **2** | → check wubu_ssm.h | Verify |
| `head_dim` (full) | **256** | → check GQA_HEAD_DIM | Verify |
| `linear_key_head_dim` | **128** | → check SSM_D_STATE | Verify |
| `linear_conv_kernel_dim` | **4** | → check CONV_KERNEL | Verify |
| `partial_rotary_factor` | **0.25** | → check if 64/256 dims used | Verify |
| `rope_theta` | **10,000,000** | → check if constant matches | Verify |
| `mrope_interleaved` | **true** | ❌ likely not implemented | Implement P2 |
| `moe_intermediate_size` | **512** | → check D_FF (should match) | ✅ Likely correct |
| `num_experts` | **256** | → check N_EXPERTS | ✅ Correct |
| `num_experts_per_tok` | **8** | → check N_ACTIVE_EXPTS | ✅ Correct |
| `vocab_size` | **248320** | → check | ✅ Correct |
| `bos_token_id` | **248044** | → check wubu_tokenizer | Verify |
| `eos_token_id` | **248044** (same!) | → check | Verify identical |

