# WuBuNesting Inference Engine — ARCHITECTURE & LEGACY

**Project**: bytropix — custom C inference engine for Qwen3.6-35B-A3B (qwen35moe)
**Author**: waefrebeorn / WuBuText AI
**Model**: Qwen3.6-35B-A3B-UD-IQ2_M.gguf (Unsloth Dynamic 2.0 quantization)
**Reference**: llama.cpp (qwen35moe.cpp via ~/llama.cpp/build/bin/llama-cli)
**Cos-sim Verified**: 0.9970 vs reference (all Phase 1-4 bugs fixed)

---

## 1. ENGINE OVERVIEW

### What It Does
Load a GGUF Qwen3.6-35B-A3B model → quantized inference on CPU only → produce coherent text output. No GPU required (GPU path experimental, MoE bottleneck limited).

### Architecture (one forward step)

```ascii
Token Embedding (Q5_K matmul)
    ↓
40× Layer Loop:
    ├── rms_norm (F32)
    ├── SSM layer (30x): attn_qkv → gate → ssm_recurrence → ssm_out → gate × residual
    │   └── MoE: router(F32) → 8/256 experts(IQ2_XXS/IQ3_XXS/Q5_K/Q6_K)
    ├── or GQA layer (10x): attn_q/k/v → IMRoPE → attention(KV cache) → output_proj
    │   └── MoE: router(F32) → 8/256 experts(...)
    └── residual_add
    ↓
rms_norm → output_proj (Q4_K: 2048×248320) → logits → softmax → token
```

### Key Stats
| Metric | Value |
|--------|-------|
| Parameters | 35B total, ~3B active per token |
| Weight size | 10.7 GB GGUF |
| Layers | 40 (30 SSM + 10 GQA) |
| Experts | 256 routed + 1 shared, 8 active |
| D_MODEL | 2048 |
| D_FF (expert) | 512 |
| Vocab | 248320 |
| Context | Unlimited (KV cache grows with seq) |
| Decode speed | 0.6-0.7 tok/s (current), target 5 tok/s |
| Cos-sim vs ref | 0.9970 |

---

## 2. FILE LAYOUT

```
/home/wubu/bytropix/
├── src/
│   ├── wubu_model.c         ← MAIN: model load, layer forward, gen_text
│   ├── wubu_gguf.c          ← GGUF reader (header, tensor, raw data)
│   ├── wubu_ssm.h           ← SSM kernel (delta-net recurrence)
│   ├── wubu_gqa.h           ← GQA kernel (grouped query attention + IMRoPE)
│   ├── wubu_moe.h           ← MoE kernel (router + quantized expert FM)
│   ├── wubu_norm.h          ← rms_norm, layer_norm
│   ├── wubu_output_proj.h   ← Output projection (Q4_K matmul)
│   ├── wubu_quantized_matmul.h  ← Quantized matmul driver (dispatch by type)
│   ├── vec_dot_generic.c    ← SIMD dot product implementations
│   └── ggml.h               ← GGML type enum, block structs
├── include/
│   ├── wubu_model.h         ← exposed API, struct defs
│   └── wubu_ssm.h           ← SSM public interface
├── tools/
│   ├── gen_text.c           ← Main text generation tool (CHAT=1)
│   ├── gen_text_gpu.c       ← GPU accelerate version (GPU=1)
│   ├── test_full_model.c    ← Single-step cos-sim verification
│   ├── test_one_layer.c     ← Layer-by-layer debugging
│   ├── dump_*.c             ← 20+ diagnostic dumper tools
│   ├── py_compare.py        ← Python reference comparison
│   └── py_*.py              ← 15+ Python verification scripts
├── Makefile                 ← Build system (make gen_text)
├── .hermes/
│   └── mind-palace/
│       ├── state.md         ← Current status
│       ├── plan.md          ← Roadmap (Phases 0-7)
│       ├── goal-mantra.md   ← Session goal paste
│       ├── prestige_prompt.md ← DA audit + next task
│       ├── overnight-map.md ← Session handoff
│       ├── plans/           ← DA audits v5-v10
│       ├── vault/
│       │   ├── bins/        ← Archived versions
│       │   ├── tmp-tools/   ← Preserved debug tools
│       │   └── deepseek-papers/ ← Research papers
│       └── research.md      ← Architecture notes
├── ~/llama.cpp/             ← Reference implementation
└── /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf ← Test model (11.5 GB)
```

---

## 3. QUANTIZATION TYPES SUPPORTED (Verified)

| GGML Type ID | Name | Used For | bpw | SIMD |
|:---:|------|----------|:---:|:----:|
| 0 | F32 | Norms, biases, routers, ssm_a/dt/beta/alpha/conv1d | 32 | — |
| 12 | Q4_K | output.weight | 5.0 | SSE/AVX2 |
| 13 | Q5_K | attn_qkv, shared gate/up, token_embd | 6.5 | SSE/AVX2 |
| 14 | Q6_K | ssm_out, shared down | 7.5 | SSE/AVX2 |
| 16 | IQ2_XXS | ffn_gate_exps, ffn_up_exps | 2.2 | C-only (needs gather) |
| 18 | IQ3_XXS | ffn_down_exps (37/40 layers) | 3.3 | C-only (needs gather) |
| 23 | IQ4_XS | ffn_down_exps (3/40 layers) | 4.3 | C-only (needs gather) |

**Critical quirk**: GGML enum values differ between codebases. Our `ggml.h` uses llama.cpp's enum:
- Q4_K=12, Q5_K=13, Q6_K=14, Q8_0=15
- IQ2_XXS=16, IQ2_S=17, IQ3_XXS=18
- IQ4_XS=23

Always verify tensor type IDs at runtime vs the enum defined in the header being compiled.

---

## 4. BUGS FOUND & FIXED (Historical Record)

### Bug 1: GQA Q/gate Interleave (May 18)
**Symptom**: Cos-sim -0.51 on GQA layers.
**Root cause**: `attn_q.weight` output [8192] is per-head interleaved as `[Q_h0][gate_h0][Q_h1][gate_h1]...`. Code was splitting as `[Q(4096)][gate(4096)]` contiguous blocks.
**Fix**: Interleave-aware load: Q get index (h*2*256 + 0..255), gate get (h*2*256 + 256..511).
**Result**: Cos-sim -0.51 → 0.9968. ALL 40 layers > 0.995.

### Bug 2: Output Proj Transpose
**Symptom**: Cos-sim -0.457 on final output.
**Root cause**: `output.weight` stored as [vs, D_MODEL] in GGUF but code assumed [D_MODEL, vs].
**Fix**: Removed transpose in matmul call.
**Result**: Cos-sim → 0.9969.

### Bug 3: SSM State Carry
**Symptom**: Multi-token decode incoherent after first token.
**Root cause**: SSM state not cached between decode steps.
**Fix**: Added `float ssm_state[N_SSM_LAYERS][D_MODEL * D_STATE]` persistent across calls.

### Bug 4: KV Cache Append-Only
**Symptom**: Decode only attended to self-position.
**Root cause**: No KV cache — single-token attention attended only the current token.
**Fix**: Buffer K_norm/V for all cache positions, compute full attention matrix each step.

### Bug 5: Tokenizer Binary Lookup
**Symptom**: Garbage tokens after position 1.
**Root cause**: Embedding file opened/closed per decode step (fopen/fclose on 2.5GB file each call).
**Fix**: Open once at init, close at cleanup.

### Bug 6: MTP Model (Will Fix)
**Status**: Not yet loaded. MTP model has 753 tensors (base has 733). Extra 20 tensors in blk.40 + nextn.*.
**Plan**: Phase 6 — add `mtp_enabled` flag, load blk.40 weights, implement MTP forward path.

---

## 5. PERFORMANCE PROFILE (16-thread CPU)

### Current Bottleneck Distribution (per decode step)

| Component | Time | % | Notes |
|-----------|:----:|:-:|-------|
| MoE (8 experts × 3 matmuls) | 12ms | 28% | Each: gate(2048×512) + up(2048×512) + down(512×2048) |
| GQA (10 layers) | 10ms | 23% | QKV proj + attention + output proj |
| SSM (30 layers) | 8ms | 19% | QKV proj + conv1d + recurrence + out proj |
| Output proj (2048×248320) | 6ms | 14% | Single Q4_K matmul, largest single op |
| Norms + router + overhead | 7ms | 16% | rms_norm × 40, router × 40 |

**Total**: ~43ms per layer × 40 layers = 1.72s → 0.58 tok/s

### Why Not Faster?
- **Memory bandwidth bound**: DDR5 ~50GB/s. 10.7GB model / 50 GB/s = 214ms minimum.
- **Current 1.72s** is 8× worse than bandwidth limit.
- **Reason**: Streaming 1 token at a time wastes bandwidth (cache-line utilization ~30-40%).

### The Island Boy Fix (Phase 5)
Process B=4 tokens per weight-load. Weight bandwidth utilization goes from 30% → 85%.
Theoretical: 214ms / 4 = 53.5ms per token → 18 tok/s (before overhead).
Realistic: 300-400ms per batch of 4 → 1.2-1.5 tok/s (after overhead).

---

## 6. KEY ARCHITECTURE NOTES

### SSM vs GQA Pattern
```
Layer pattern: [SSM, SSM, SSM, GQA] × 10
SSM layers: 0,1,2, 4,5,6, 8,9,10, ... (30 total)
GQA layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 (10 total)
```

SSM layers have: `attn_qkv` (fused Q/K/V) + `attn_gate` (sigmoid gate) + SSM recurrence.
GQA layers have: separate `attn_q`, `attn_k`, `attn_v` (no gate) + IMRoPE + attention.

### MoE Expert Layout
```
ffn_gate_exps:  [D_MODEL=2048, D_FF=512, N_EXPERTS=256]  IQ2_XXS (2.2bpw)
ffn_up_exps:    [2048, 512, 256]  IQ2_XXS
ffn_down_exps:  [512, 2048, 256]  IQ3_XXS (37L) or IQ4_XS (3L)

Shared expert (always activated):
ffn_gate_shexp:  [2048, 512]  Q5_K  (6.5bpw)
ffn_up_shexp:    [2048, 512]  Q5_K
ffn_down_shexp:  [512, 2048]  Q6_K  (7.5bpw)
```

**dims[0] is innermost** (GGML convention). Per-expert byte offset = `gguf_raw_size(type, D_MODEL*D_FF)` for gate/up, `gguf_raw_size(type, D_FF*D_MODEL)` for down.

### IMRoPE (Qwen3.6)
- rope.dimension_sections = [11, 11, 10, 0]
- Vectors: Qh_dim=256 split into freq_dim=11, other_dim=11, other_dim=10, null_dim=0
- Actually freq_dim=32 total: 11+11+10 = 32 used, last 0 unused
- theta = 10_000_000 (10M)

---

## 7. QUANTIZED MATMUL PATH

### For K-quant types (Q4_K, Q5_K, Q6_K):
```
void* type_weight = raw_gguf_data + d_meta_offset + d_quant_offset;

// 1. SIMD vec_dot (SSE2/AVX2)
// 2. Accumulate block sums
// 3. Apply min/max scale
// 4. Result: F32 dot product

dispatch: proj_matmul(float* out, float* in, type_weight, D_IN, D_OUT, type)
    → calls vec_dot_generic.c: dot_type_{q4k,q5k,q6k}(src0[i], src1[j], ...)
```

### For IQ types (IQ2_XXS, IQ3_XXS, IQ4_XS):
**No SIMD yet**. Generic C path using lookup tables:
- IQ2_XXS: 2-bit → 8192-entry grid → 4 weights per byte
- IQ3_XXS: 3-bit → 8192-entry grid + 512-entry ksigns → byte-aligned packing
- IQ4_XS: 4-bit → 8192-entry grid + scaling → block structure

### Output Projection (Q4_K, 2048×248320):
Largest single operation: ~500M FMAs. Uses `proj_matmul` with dot_type_q4k.
Currently ~6ms per forward. Already fast enough (<14% of time).

---

## 8. BINARY TOOLS LEGACY

| Tool | Purpose |
|------|---------|
| `gen_text` | Main text generator (CHAT=1 env var for Qwen chat template) |
| `test_full_model` | Cos-sim verification against llama.cpp reference |
| `test_one_layer` | Per-layer dump + compare |
| `ref_dumper` | Fast reference outputs via libllama.so |
| `diff_models` | Compare tensor sets between model variants |
| `dump_tensor_types` | List all tensors with GGML type IDs |
| `dump_layer_N` | Reference hidden state dump for layer N |
| `py_compare.py` | Python diff: bytropix vs reference outputs |
| `py_layer_stats.py` | Statistical analysis of layer outputs |

All tools built via `make <toolname>`.

---

## 9. MTP MODEL (Future)

**Model**: `/models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf` (11.9 GB, 753 tensors)
**Extra layer**: blk.40 (GQA + MoE) with `nextn.*` head
**Tensors**:
- `blk.40.*`: Same structure as layers 0-39 (GQA type)
- `nextn.hnorm.weight` → hidden norm before projection
- `nextn.enorm.weight` → embedding norm
- `nextn.eh_proj.weight` → hidden→vocab projection for draft logits
- `nextn.shared_head_norm.weight` → shared head norm

**Purpose**: Self-drafting speculative decoding. blk.40 predicts several tokens cheaply, then full 40-layer model verifies in batch. Expected 2-3× speedup.

---

## 10. BUILD & RUN

```bash
# Build
cd /home/wubu/bytropix
make gen_text         # CPU inference
make gen_text_gpu     # GPU offloaded (experimental)
make test_full_model  # Cos-sim test

# Run
./gen_text "The capital of France is" 32          # 32 tokens
CHAT=1 ./gen_text "Hello" 128                      # Chat mode
GPU=1 ./gen_text "The capital of" 64               # GPU output proj
PROFILE=1 ./gen_text "Test" 1                      # Per-layer timing

# Verify
./test_full_model                                   # Compare vs ref
./ref_dumper "The capital of"          # Get reference output
python3 tools/py_compare.py ref_dump our_dump       # Python diff
```

---

## 11. RESEARCH PAPERS CONSULTED

### Vault (.hermes/vault/)
1. **DeepSeek-V3 Technical Report** — MoE architecture, MTP self-drafting, MLA attention
2. **DeepSeekMoE** — Fine-grained expert segmentation, shared experts, normalized sigmoid gating
3. **DeepSeekMoE Statistical** — Theoretical justification for shared experts + sigmoid gating
4. **Gemma 2/3 Technical Reports** — Reference transformer architectures
5. **Unsloth Dynamic 2.0 Quant Formula** — Per-tensor quantization analysis (Qwen3.6-35B-A3B-UD)

### Key Insights Applied
- 256 experts / 8 active matches DeepSeek-V3's MoE configuration exactly
- MTP for speculative decoding without a separate draft model (DeepSeek-V3 §4)
- Normalized sigmoid gating for MoE router (from DeepSeekMoE paper)
- Memory bandwidth is the bottleneck, not compute (confirmed by profiling)

---

## 12. MIND PALACE INDEX

```
.hermes/mind-palace/
├── state.md              ← CURRENT STATUS (always up-to-date)
├── plan.md               ← ROADMAP Phases 0-7 (triple-extended)
├── goal-mantra.md        ← Session goal paste
├── prestige_prompt.md    ← DA audit + next task
├── overnight-map.md      ← Session handoff summary
├── research.md           ← Architecture reference
├── entry.md              ← First entry point
├── index.md              ← Navigation
├── README.md             ← How to use mind palace
├── project.md            ← Project overview
├── testing.md            ← Testing methodology
├── plans/
│   ├── devils_advocate_v10.md  ← Latest (10 gaps, all CLOSED)
│   ├── devils_advocate_v9.md   ← Previous
│   ├── devils_advocate_v{5-8}.md
├── tier1-core/
│   └── 2-arch-reference/README.md  ← Architecture reference
└── vault/
    ├── bins/              ← Archived old versions (by timestamp)
    ├── tmp-tools/         ← Preserved debug tools
    ├── deepseek-papers/   ← Saved architecture papers
    └── synthesis.md       ← Cross-paper synthesis
```

---

*Legacy document — last updated May 18, 2026*
*Engine: bytropix — Qwen3.6-35B-A3B custom C inference*
*All bugs fixed as of Phase 4. Phase 5+ under development.*
*"Island boy batch decode, MTP spec-deck, hardware saturate."*
