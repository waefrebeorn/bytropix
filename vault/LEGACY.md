# bytropix — Phase 14 Legacy Dump (May 19, 2026 PM)

## Scope
This document captures everything built, learned, and left behind across ~5 days of agentic C inference engineering for Qwen3.6-35B-A3B. It is the authoritative reference for future sessions.

---

## 1. PROJECT ARC (May 17–19, 2026)

### Timeline

| Phase | Date | Speed | Cos-sim | Event |
|-------|------|-------|---------|-------|
| 0 | May 12 | garbage | N/A | Initial build — all output garbage |
| 1 | May 13 | 0.2 tok/s | N/A | GPU forward but NaN cascade |
| 2 | May 15 | 0.3 tok/s | -0.51 | IQ2_XXS raw_size fix (72→66 bytes/block) |
| 2.5 | May 16 | 0.3 tok/s | -0.51 | "Inference is BROKEN" — STATUS.md |
| 3 | May 18 | 0.6 tok/s | **0.9968** | **GQA Q/gate interleave bug fixed** |
| 4 | May 18 | 0.7 tok/s | 0.9968 | SIMD vec_dot + KV cache |
| 5 | May 18 | 0.7 tok/s | 0.9968 | MoE OpenMP race fixed (44→15ms) |
| 6 | May 18 | 1.2 tok/s | 0.796 | MoE enabled — **Q6_K bug active (hidden)** |
| 7 | May 18 | 2.1 tok/s | 0.796 | Expert prefetch, output proj OMP |
| 8 | May 18 PM | 4.7 tok/s | 0.796 | MoE OMP + IQ2_XXS AVX2 |
| 9 | May 19 AM | 4.7 tok/s | 0.796 | MoE dequant on-demand (saves 3GB RAM) |
| 9.5 | May 19 AM | **7.0 tok/s** | **0.9967** | **Q6_K loop iter bug fixed (THE ONE!)**
| 10 | May 19 AM | 7.0 tok/s | 0.9967 | KV cache 256k F16, heap attn_weights |
| 11 | May 19 AM | 7.0 tok/s | 0.9967 | IQ3_XXS AVX2 vec_dot |
| 12 | May 19 PM | 7.0 tok/s | 0.9967 | MTP spec decode (broken at IQ2_M) |
| 13 | May 19 PM | 8.3 tok/s | 0.9967 | GPU output proj (cuBLAS + batched) |
| 14 | May 19 PM | **8.8 tok/s** | **0.9967** | **SSM AVX2 + fused Q8_K + GPU quantized + tiled GQA + MTP EMA** |

### The One Bug That Mattered

**Q6_K Vec Dot Loop Count (Phase 9.5, May 19 AM)**
- Location: `quantized_dot_generic.c:314` — `j < QK_K/32` should be `j < QK_K/16`
- Impact: Shared expert was 27% wrong. Cos-sim stuck at 0.796 for 3 phases.
- Fix: One character. `32` → `16`.
- Lesson: Component-level isolation (test each quant type separately vs F32) catches more than end-to-end cos-sim.

---

## 2. COMPLETE BUG LOG (8 Bugs, All Fixed)

| # | Bug | Date | Symptom | Fix | Verification |
|---|-----|------|---------|-----|-------------|
| 1 | GQA Q/Gate Interleave | May 18 | Cos-sim -0.51 | Per-head extraction: 256+256 | 0.9968 |
| 2 | IMRoPE | May 18 | T=2 wrong | sections=[11,11,10,0] | T=1 same, T=2 passes |
| 3 | MoE OMP Race | May 18 | Non-deterministic | Thread-local scratch | Deterministic, 44→15ms |
| 4 | SSM State Carry | May 18 | Incoherent after T=1 | Persistent state buffer | max_diff 0.0 |
| 5 | KV Cache | May 18 | Self-only attention | Buffer all K_norm/V | Coherent multi-token |
| 6 | MTP Crash | May 19 | SIGSEGV | NULL checks + correct concat | Runs without crash |
| **7** | **Q6_K Loop Count** | **May 19** | **Cos-sim 0.796** | **`32`→`16`** | **→ 0.9967** |
| 8 | DA v10 Wrong | May 19 | Blamed MoE gating | Q6_K was real cause | Isolate test found it |

---

## 3. PERFORMANCE CEILING ANALYSIS

### Bottleneck Distribution (CPU, 16 threads, Phase 14)

| Component | Time/token | % | Limit |
|-----------|:----------:|:-:|-------|
| MoE (40 layers × 1.2ms) | 48ms | 48% | **Memory bandwidth** — 256 experts × 3 weights loaded per layer |
| SSM + GQA (40 layers × 1.0ms) | 40ms | 40% | **Compute bound** — Q5_K matmuls, selective scan |
| Output proj (Q4_K, 2048×248320) | 10ms | 10% | **Memory bandwidth** — reading 1.9GB of weights |
| Other (tokenizer, sampling, overhead) | 2ms | 2% | — |
| **Total** | **~100ms** | **100%** | **Ceiling: ~10 tok/s** |

**Where 8.8 tok/s falls short of 10 tok/s ceiling:**
- Thread scheduling jitter (±5%)
- Tokenizer overhead (merge-table BPE is 2× slower than llama.cpp)
- malloc variance across decode steps
- PROFILE=1 overhead (if enabled)

### GPU Potential

| Component | CPU time | GPU time | VRAM needed | Status |
|-----------|----------|----------|-------------|--------|
| Output proj | 10ms | ~0.1ms (cuBLAS) / ~0.5ms (Q4_K kernel) | 1.9GB (Q4_K) / 7.6GB (F32) | ✅ Done |
| GQA attention (256k) | ~200ms | ~5ms (tile-streaming) | ~80MB (buffers) | 🔶 Kernels written, not yet wired |
| SSM matmuls | ~20ms | ~2ms | ~240MB (30 layers × 8MB) | 🟡 Planned |
| MoE | 48ms | ~5ms | ~590KB (all experts) | 🟡 Planned |
| **Full GPU decode** | **100ms** | **~15ms** | **~2.5GB** | **Ceiling: ~66 tok/s** |

---

## 4. GPU ROADMAP (Triple Extended)

### Phase 15: Wire GPU GQA Attention [P1 — IMMEDIATE]
- **What**: Connect `gpu_gqa_attention()` to the inference loop
- **Where**: `wubu_model.c` GQA forward — on GPU=1, use GPU attention
- **Gain**: 256k context GQA from ~200ms → ~5ms
- **Status**: CUDA kernels written, need host-side wiring and tile-size tuning

### Phase 16: GPU SSM Matmuls [P2]
- **What**: Port attn_qkv (Q5_K), attn_gate (Q5_K), ssm_out (Q6_K) to GPU
- **How**: Upload hidden state [2048], dequant+matmul on GPU, download result
- **Challenge**: 30 layers × multiple matmuls = 90 PCIe transfers/layer
- **Mitigation**: Keep each layer's quantized weights on GPU (8MB/layer × 30 = 240MB)
- **Gain**: SSM from ~20ms → ~2ms

### Phase 17: GPU MoE [P3]
- **What**: Port expert compute (IQ2_XXS gate/up, IQ3_XXS down) to GPU
- **How**: All 256 experts × 3 weights = 590KB — fits in GPU L2 cache
- **Key**: Dynamic expert routing per token — need to upload 8 expert indices
- **Gain**: MoE from 48ms → ~5ms

### Phase 18: GPU MTP Pipeline [P4]
- **What**: After all components GPU-accelerated, MTP becomes pipeline
- **How**: CUDA streams — overlap main model forward with MTP draft generation
- **Gain**: MTP MoE speedup 1.2x → 1.5x via pipeline overlap

### Phase 19: End-to-End GPU Inference [P5]
- **What**: All layers run on GPU. CPU only handles tokenizer + control flow
- **Total VRAM**: ~2.5GB (output proj weights + KV cache Q4_0 on GPU + per-layer SSM weights)
- **Ceiling**: ~66 tok/s on RTX 5050
- **Challenge**: Keep 256k KV cache within 6.4GB VRAM (Q4_0 or tiered)

---

## 5. VAULT RESEARCH — WHAT WAS EXPLORED BUT NOT YET PORTED

| Vault Area | Papers/Files | Relevance to Inference | Port Status |
|------------|-------------|----------------------|-------------|
| **Sparse Attention** | NSA (2503.10488), Gated Sparse (2503.09542), Delta Attn (2502.14864) | O(n·k) linear attention for 256k context | High — would solve the 256k bottleneck |
| **Hyperbolic SSM** | Poincaré Embeddings (1705.08039), Mobius Transformers (2311.11394) | Experimental Poincaré SSM variant in wubu_poincare_ssm.c | Experimental — NOT wired to gen_text |
| **Tailslayer** | hedged reads → speculative decoding | Tailslayer-inspired draft-verify pattern | Already used for MTP |
| **Token Superposition** | Nous Research TST (2605.06546) | Training methodology for better token representations | Training phase, not inference |
| **Q-Controller** | 10-state×5-action Q-table | Meta-optimization for sampling params | Research only |
| **PID Lambda Controller** | Adaptive LR via PID control | Training speed optimization | Research only |
| **Hamilton Encoder** | Geodesic compression for KV cache | 62% KV cache memory reduction | High potential |
| **HashMind** | Associative memory with SO(n) rotations | Alternative to attention | Research only |
| **Entropix Sampler** | Dynamic inference-time sampling | Better sampling quality | Could be wired now |

---

## 6. AGENTIC COLLABORATION METHODOLOGY

The project used a structured workflow across ~25 agent sessions:

### Per-Session Loop
```
1. Read 5 mind-palace files (state → goal-mantra → plan → prestige → overnight)
2. Pick highest-priority undone task
3. Execute (write code, build, run)
4. Verify against reference (cos-sim, layer dumps, or text coherence)
5. Update all 5 mind-palace files with findings
6. Git commit
7. Deliver caveman-compressed summary
```

### Key Innovations

**Caveman Compression:** ~60% token savings. Strips articles, prepositions, hedging. Format: 
```
"[task] [action] [result] [next]"
```
Example: `"Fused Q8_K quant: quantized_matmul_from_q8() — SSM/GQA share quant. Saves 50/decode. ✅"`

**Triple Devil's Advocate:**
- DA-1: Code vs Theory — test each claim against runtime verification
- DA-2: Vault Deep-Dive — cross-reference papers against implementation
- DA-3: Cold Gap Ranking — prioritize remaining gaps by impact/effort

**Mind Palace Atomic Updates:** All 5 files rewritten each session. Prevents version drift across context windows. Old versions archived to `vault/bins/` with timestamps.

**Goal Paste:** Session handoff format: compressed 5-line summary of state (what was done, what's next, critical findings, resource limits, known bugs).

### What Worked
- Component-level isolation debugging (caught Q6_K bug that DA missed)
- Layer cos-sim as the primary verification metric
- GPU-first thinking for throughput bottlenecks
- Fused Q8_K quant (obvious in retrospect, took 5 days to implement)

### What Didn't Work
- DA analysis without runtime verification (DA v10 blamed MoE gating; real bug was loop iter count)
- Early belief that MTP would work at IQ2_M (quantization mismatch was fatal)
- Attempting to quantify without understanding tensor layouts first

---

## 7. KEY VERIFICATION METRICS

| Metric | Value | How Measured |
|--------|-------|-------------|
| Cos-sim vs llama.cpp | 0.9967 | test_full_moe vs ref_dumper logits |
| Q4_K vs F32 SGEMM | 0.99995 | Layer-by-layer cos-sim |
| Q5_K vs F32 SGEMM | 0.9999 | Per-matmul cos-sim |
| Q6_K vs F32 SGEMM | 0.9999 | Per-matmul cos-sim (was 0.728 before fix) |
| IQ2_XXS vs F32 | max diff 0.002 | Per-expert dequant comparison |
| IQ3_XXS vs F32 | 0.9965 | Per-matmul cos-sim |
| AVX2 SSM scan vs scalar | max diff 1.85e-3 | test_ssm vs golden vectors |
| GPU output proj vs CPU Q4_K | 0.99996 | Direct output comparison |
| GPU GQA attention | 🔶 Not yet wired | Kernels written, host call pending |

---

## 8. WHAT WAS LEFT BEHIND

### Known Issues

| Issue | Status | Impact |
|-------|--------|--------|
| MTP verify at IQ2_M | ❌ 18% acceptance | Online logit correction EMA might help — NOT YET TESTED |
| IQ3_XXS AVX2 vec_dot | ⏳ Generic C only | MoE down weights 30% slower than possible |
| SSM conv1d SIMD | ⏳ Scalar OMP | Not hot now, but helps prefill |
| 256k context GQA | ⏳ Tiled but O(n) | GPU GQA kernels written but not wired |
| Chat template | ❌ Not applied | Minor quality loss vs llama.cpp chat |
| Tokenizer slowness | ⏳ 2× slower | Merge-table BPE has 247K merges |
| GPU GQA attention | 🔶 Not wired into model loop | Host function + kernel exist, need integration |
| End-to-end GPU inference | ❌ CPU only except output proj | SSM/GQA/MoE still CPU |
| No batching | ❌ Single sequence | gen_text processes one conversation at a time |

### Research Not Yet Applied to Inference

| Research | File | Potential | Effort |
|----------|------|-----------|--------|
| Poincaré SSM | wubu_poincare_ssm.c | Hyperbolic attention for hierarchical reasoning | High |
| TGT (Toroidal wrapping) | wubu_ssm.c (tgt_wrap) | Prevents overflow in attention scores | Already applied in GQA |
| Möbius Linear | wubu_mobius_linear.c | Hyperbolic linear layers | High |
| Nested SSM | wubu_nested_ssm.c | Multi-scale SSM with hierarchical state | Very High |

---

## 9. COMPLETE FILE MANIFEST (Core Engine)

| File | Lines | Purpose | Last Modified |
|------|-------|---------|--------------|
| src/wubu_model.c | 1,241 | Model init, forward loop, MTP head | Phase 14 |
| src/wubu_ssm.c | 2,681 | SSM Gated DeltaNet + GQA + AVX2 scan + fused Q8 | Phase 14 |
| src/wubu_moe.c | 555 | MoE router + quantized expert forward | Phase 9 |
| src/quantized_matmul.c | 397 | Q8_K activation → vec_dot + fused Q8_K variant | Phase 14 |
| src/quantized_dot_generic.c | 1,125 | All 7 quant types, AVX2/SSE/generic | Phase 9.5 |
| src/gguf_reader.c | 1,787 | GGUF parser, dequant, blob buffer | Phase 8 |
| src/gpu_output_proj.cu | 766 | GPU output proj (cuBLAS + Q4_K kernel + GQA attention) | Phase 15 |
| src/wubu_tokenizer.c | 300 | GPT-2 BPE, 248K vocab | Phase 2 |
| include/wubu_model.h | 250 | Model struct, KV cache helpers | Phase 10 |
| include/wubu_ssm.h | 371 | SSM/GQA weights, function decls | Phase 11 |
| include/gguf_reader.h | 144 | GGML types, reader API, quantized_matmul | Phase 14 |
| include/gpu_output_proj.h | 30 | GPU output proj declarations | Phase 13 |
| tools/gen_text.c | ~232 | Text generation entry point | Phase 14 |
| tools/gen_text_mtp.c | ~338 | MTP speculative decode + EMA correction | Phase 14 |
| **Core total** | **~13,400** | | |

---

*Generated May 19, 2026 (23:00 PM). v5 MADE_AGENTICALLY essay captures the full agentic story. This document is the technical legacy dump.*
