# bytropix — Triple Extended GPU Roadmap & Legacy Plan (May 19, 2026 PM)

## Phase 0-14: DONE ✅
| Phase | Detail | Speed | Cos-sim | Status |
|-------|--------|-------|---------|--------|
| 0-11 | GQA attn, vec_dot, MoE, KV cache, quant path | 7.0 | 0.9967 | ✅ |
| 12 | MTP Spec Decode (two-model load, DRAFT_N=2) | 7.0 | 0.9967 | ✅ MTP broken at IQ2_M |
| 13 | GPU Output Proj (cuBLAS + batched) | 8.3 | 0.9967 | ✅ |
| 14 | SSM AVX2 + fused Q8_K + GPU quantized + tiled GQA | 8.8 | 0.9967 | ✅ |

---

## GPU ROADMAP — Triple Extended (Phases 15-19)

### Phase 15: Wire GPU GQA Attention [P1 — IMMEDIATE]
**Goal:** Connect GPU GQA tile-streaming kernels to the inference loop
**Speed target:** 256k GQA from ~200ms → ~5ms
**Kernels (written, need wiring):**
- `gqa_qk_kernel` — Q·K dot product, one block per Q head per KV position
- `gqa_find_max_kernel` — per-head max score reduction
- `gqa_sumexp_kernel` — per-head softmax denominator reduction
- `gqa_softmax_v_kernel` — V weighted sum with atomicAdd
- `gpu_gqa_attention()` — host orchestrator for tile streaming

**Wiring needed:**
- `wubu_model.c`: On `GPU=1`, upload Q to GPU, call `gpu_gqa_attention()`, download result
- Swap into existing attention path: replace CPU Q·K loop with GPU call
- **Gain:** ~8× speedup at 256k context

### Phase 16: GPU SSM Matmuls [P2]
**Goal:** Port attn_qkv (Q5_K, 2048×8192), attn_gate (Q5_K, 2048×4096), ssm_out (Q6_K, 4096×2048)
**Speed target:** SSM from ~20ms → ~2ms
**Approach:**
1. Keep each layer's quantized weights on GPU (8MB/layer × 30 = 240MB total)
2. Adapt GPU_QUANTIZED-style Q5_K/Q6_K dequant kernel for SSM matmuls
3. Upload hidden state [2048], run dequant+matmul on GPU, download result [C=8192 or 4096]
4. Pipeline across layers: upload next layer's hidden state while current layer computes

**VRAM per layer:** 8MB Q5_K weights + 0.5MB Q6_K weights = 8.5MB × 30 ≈ 255MB
**Challenge:** 30 layers × 3 matmuls = 90 separate kernel launches
**Mitigation:** Batch all 3 matmuls for one layer into a single kernel

### Phase 17: GPU MoE Expert Compute [P3]
**Goal:** Port 8-expert IQ2_XXS/IQ3_XXS compute to GPU
**Speed target:** MoE from 48ms → ~5ms
**Approach:**
1. Keep all 256 experts × 3 weight types on GPU (590KB total — fits in L2!)
2. Upload hidden state [2048], router output (8 expert indices, 8 weights)
3. GPU kernel loads 8 experts' weights from device memory, computes gate*up→silu→down
4. Download result [2048]

**Key insight:** 256 experts at IQ2_XXS gate up = 66 bytes/block × 8 blocks × 256 × 2 = 270KB. IQ3_XXS down = 98 bytes/block × 2 blocks × 256 = 50KB. Total: ~590KB for ALL 256 experts' weights. This fits in GPU L2 cache (1MB+ on RTX 5050).

**VRAM:** ~590KB for weights + 16KB for scratch = negligible

### Phase 18: GPU MTP Pipeline [P4]
**Goal:** After full GPU decode, overlap MTP draft + verify on CUDA streams
**Speed target:** MoE MTP speedup from 1.2× to 1.5× via pipeline
**Approach:**
1. CUDA Stream 0: Main model forward (GPU layers)
2. CUDA Stream 1: MTP draft generation (blk.40 on GPU, overlaps with stream 0)
3. Both streams synchronize for verify step
4. Pipeline depth: draft[0] overlaps with main[0], draft[1] overlaps with main[1], etc.

**Expected decode pipeline:**
```
Token T:  [main model forward] → [emit token] → [draft[0]] → [draft[1]]
Token T+1:                    [verify] → [emit draft] → [main forward] → ...
```
Pipeline can overlap draft generation with verify from previous step.

### Phase 19: End-to-End GPU Inference [P5]
**Goal:** All layers run on GPU. CPU only handles tokenizer + control flow
**Speed target:** ~66 tok/s on RTX 5050
**VRAM budget (6.4GB):**
- Output proj: 1.9GB (Q4_K mode)
- SSM weights (30 layers): 255MB
- GQA weights (10 layers): 85MB
- MoE experts: ~1MB
- KV cache: depends on compression
- **Total model:** ~2.3GB
- Remaining: ~4GB for buffers + KV cache

**KV cache at 256k:** 256k × 2 × 256 × 2 = 256MB per cache type × 10 layers = 5GB → doesn't fit.
**Solution:** Q4_0 KV cache = 256k × 2 × 256 × 0.5 = 64MB per layer × 10 = 640MB → fits.

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

### Training Pipeline
Not yet started. Would need:
1. GPU backward pass kernels for SSM (wubu_ssm_backward exists)
2. GPU backward pass kernels for MoE (wubu_moe_backward exists)
3. GPU gradient accumulation + optimizer (RSGD exists)
4. Training loop: forward → loss → backward → optimizer step

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

## KEY METRICS (Phase 14 Verified)

| Metric | Value | Verification Method |
|--------|-------|-------------------|
| Cos-sim vs llama.cpp | 0.9967 | test_full_moe vs ref_dumper |
| Decode speed (CPU) | 8.8 tok/s | gen_text 10 tok decode, 16 threads |
| Decode speed (GPU quantized) | 9.4 tok/s | gen_text_gpu GPU_QUANTIZED=1 |
| Prefill speed | 13.1 tok/s | 5-token prompt |
| Output proj (CPU) | ~10ms | PROFILE=1 |
| Output proj (GPU cuBLAS) | ~0.1ms | CUDA event timing |
| Output proj (GPU Q4_K) | ~0.5ms | CUDA event timing |
| SSM per layer | ~1.0ms | PROFILE=1 |
| GQA per layer (short ctx) | ~0.8ms | PROFILE=1 |
| MoE per layer | ~1.2ms | PROFILE=1 |
| KV cache memory | 5 GB | 256k × 10 × 2 × 2 bytes |
| GPU weight (Q4_K mode) | 1.9 GB | Output proj only |
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
