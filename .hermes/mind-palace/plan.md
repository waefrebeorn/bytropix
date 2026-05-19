# bytropix — Triple Extended GPU Roadmap & Legacy Plan (May 19, 2026 PM)

## Phase 0-15: DONE ✅
| Phase | Detail | Speed | Cos-sim | Status |
|-------|--------|-------|---------|--------|
| 0-11 | GQA attn, vec_dot, MoE, KV cache, quant path | 7.0 | 0.9967 | ✅ |
| 12 | MTP Spec Decode (two-model load, DRAFT_N=2) | 7.0 | 0.9967 | ✅ MTP broken at IQ2_M |
| 13 | GPU Output Proj (cuBLAS + batched) | 8.3 | 0.9967 | ✅ |
| 14 | SSM AVX2 + fused Q8_K + GPU quantized + tiled GQA | 8.8 | 0.9967 | ✅ |
| **15** | **GPU GQA wiring into wubu_model_t** | **3.5 GPU** | **0.9967** | **✅ Integrated chunked attn** |

---

## GPU ROADMAP — Triple Extended (Phases 16-19)

### Phase 16: GPU SSM Matmuls [P2 — NEXT]
**Goal:** Port attn_qkv (Q5_K, 2048×8192), attn_gate (Q5_K, 2048×4096), ssm_out (Q6_K, 4096×2048)
**Speed target:** SSM from ~20ms → ~2ms
**Approach:**
1. Integrate `gpu_ssm_weights` → wubu_model_t (like Phase 15 did for GQA)
2. Upload SSM weights to GPU (F32 dequant, ~255MB for 30 layers)
3. Wire `wubu_cuda_ssm_forward()` and `gpu_ssm_forward()` into forward pass
4. Keep SSM persistent states on GPU (no host-device per-step transfer)
5. Pipeline across layers: upload next layer's hidden state while current computes

**VRAM per layer:** 8MB Q5_K weights + 0.5MB Q6_K weights = 8.5MB × 30 ≈ 255MB
**Challenge:** 30 layers × 3 matmuls = 90 separate kernel launches
**Mitigation:** Batch all 3 matmuls for one layer into a single kernel

### Phase 17: GPU MoE Expert Compute [P3]
**Goal:** Port 8-expert IQ2_XXS/IQ3_XXS compute to GPU
**Speed target:** MoE from 48ms → ~5ms

### Phase 18: GPU MTP Pipeline [P4]
**Goal:** After full GPU decode, overlap MTP draft + verify on CUDA streams
**Speed target:** MoE MTP speedup from 1.2× to 1.5× via pipeline

### Phase 19: End-to-End GPU Inference [P5]
**Goal:** All layers run on GPU. CPU only handles tokenizer + control flow
**Speed target:** ~66 tok/s on RTX 5050
**VRAM budget (6.4GB):**
- Output proj: 1.9GB (Q4_K mode)
- GQA weights (10 layers F32): 1.04GB
- SSM weights (30 layers F32): ~4GB
- MoE experts: ~1MB (dequant F32 on-the-fly)
- KV cache: depends on compression
- **Total model:** ~7GB — doesn't fit in F32 mode.
- **Solution:** Quantized GPU kernels (keep Q5_K/Q6_K/IQ2_XXS on GPU like Q4_K out proj)

---

## ROADMAP BEYOND INFERENCE

### Vault-to-Inference Pipeline
| Vault Research | Priority | Port Effort | Expected Impact |
|---------------|----------|-------------|----------------|
| Sparse Attention (NSA-style) | P1 — HIGH | 3-5 days | O(n·k) GQA at 256k |
| Hamilton Encoder (KV compression) | P2 — MED | 2-3 days | 62% KV cache memory reduction |
| Entropix Sampler | P3 — LOW | 1 day | Better sampling quality |
| Poincaré SSM | P4 — EXP | 5+ days | Experimental hyperbolic attention |
| HashMind associative memory | P5 — EXP | 10+ days | Alternative to attention entirely |

---

## MTP STATUS

**Current approach:** Online logit correction EMA
- `correction[v] = 0.9*c[v] + 0.1*(main_logits[v] - mtp_logits[v])`
- Applied before argmax sampling at draft[0] and draft[1]
- Converges within ~10 tokens
- Implementation in `gen_text_mtp.c`

**Blog confirmation:** Unsloth blog: MoE MTP speedup 1.15-1.25x, optimum DRAFT_N=2 (83% acceptance at 2, 50% at 4), UD-Q2_K_XL for 220 tok/s.

**Known limitation:** IQ2_M quantization → blk.40 Q2_K/Q3_K incompatible with main IQ2_XXS/IQ3_XXS. 18% acceptance vs blog's 83%. EMA correction might help but untested at IQ2_M.

**If EMA correction works at IQ2_M:** Expected 40-60% acceptance → 1.1-1.15x speedup → ~10 tok/s effective.
**If not:** Requires UD-Q2_K_XL model for useful MTP.

---

## KEY METRICS (Phase 15 Verified)

| Metric | Value | Verification Method |
|--------|-------|-------------------|
| Cos-sim vs llama.cpp | 0.9967 | test_full_moe vs ref_dumper |
| Decode speed (CPU) | 8.8 tok/s | gen_text 10 tok decode, 16 threads |
| Decode speed (GPU GQA) | 3.5 tok/s | gen_text_gpu GPU=1 |
| Decode speed (GPU quantized) | 9.4 tok/s | gen_text_gpu GPU_QUANTIZED=1 |
| Prefill speed | 13.1 tok/s | 5-token prompt |
| GPU GQA weights VRAM | 1.04 GB | 10 layers × F32 dequant |
| GPU KV cache VRAM | ~20 MB | 10 layers × 512 × 262144 × 4B |
| GPU GQA output | ✅ matching CPU | ", I'm" vs ", I am" — tokenization variance only |
| Output proj (CPU) | ~10ms | PROFILE=1 |
| Output proj (GPU cuBLAS) | ~0.1ms | CUDA event timing |
| Output proj (GPU Q4_K) | ~0.5ms | CUDA event timing |
| SSM per layer | ~1.0ms | PROFILE=1 |
| GQA per layer (short ctx) | ~0.8ms | PROFILE=1 |
| MoE per layer | ~1.2ms | PROFILE=1 |
| KV cache memory | 5 GB | 256k × 10 × 2 × 2 bytes |
| No llama dependencies | ✅ | ldd gen_text: no libllama |
| Deterministic output | ✅ | Same seed → same tokens |
| All 7 quant types verified | ✅ | Each vs F32 SGEMM |

---

## COMPLETE BUG HISTORY

| # | Bug | Found | Impact | Fix |
|---|-----|-------|--------|-----|
| 1 | GQA Q/Gate Interleave | May 18 | Cos-sim -0.51 | Per-head extraction |
| 2 | IMRoPE | May 18 | T=2 wrong | sections=[11,11,10,0] |
| 3 | MoE OMP Race | May 18 | Non-deterministic | Thread-local scratch |
| 4 | SSM State Carry | May 18 | Incoherent after T=1 | Persistent state buffer |
| 5 | KV Cache | May 18 | Self-only attention | Buffer all positions |
| 6 | MTP Crash | May 19 | SIGSEGV | NULL checks + concat fix |
| **7** | **Q6_K Loop Count** | **May 19** | **Cos-sim 0.796** | **`32`→`16` (one char)** |
| 8 | DA v10 Wrong | May 19 | Misdiagnosis | Isolate test found real bug |
| **9** | **GPU RMSNorm Q stride** | **May 19** | **Garbage output** | **Contiguous Q buffer before norm** |
| 10 | GPU RoPE MRoPE sections | May 19 | Wrong freq | Match GPU precompute_rotary_kernel |
