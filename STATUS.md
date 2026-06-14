# STATUS — Multi-Model Integration (June 14, 2026)

**🔁 Three-model benchmark target: DiffusionGemma-26B + Gemma 4 12B QAT + Qwen3.6-35B**

---

## Current Status

| Asset | Status | Notes |
|-------|--------|-------|
| DiffusionGemma-26B GGUF | ✅ Downloaded | Q4_K_M, 16.8 GB |
| Gemma 4 12B QAT GGUF | ✅ Downloaded | Q4_K_XL, 6.7 GB |
| Qwen3.6-35B GGUF | ✅ Downloaded | IQ2_M |
| Multi-model adapter | ✅ Working | Auto-detects naming convention |
| Dynamic dimension extraction | ✅ Working | d_model from GGUF tensor shapes |
| Dynamic KV cache layout | ✅ Working | Per-layer offsets, variable kv_dim |
| Build (gen_text_cpu) | ✅ Compiles | Only warnings, no errors |
| DiffusionGemma model load | ✅ Working | All 30 layers GQA, correct dims |
| DiffusionGemma forward | ❌ Crashes | Per-layer head_dim mismatch (LARGE layers) |
| Gemma 4 12B forward | ⏳ Not tested | Dedicated engine in wubu_gemma4.c |
| Qwen3.6-35B forward | ✅ Working | 3-4 tok/s CPU, coherent output |
| 512K benchmark (all 3) | ⏳ Pending | Blocked on DGemma forward crash |

---

## Multi-Model Adapter Status

### Architecture Detection

| Model | Naming | Detection | is_ssm Logic |
|-------|--------|-----------|---------------|
| Qwen3.6 | `blk.%d.*` | `tensor_naming=0` | `(layer_idx+1)%4 != 0` → 30 SSM, 10 Gemma |
| Gemma 4 | `model.layers.%d.*` | `tensor_naming=1` | All GQA (0 SSM) |
| DiffusionGemma | `model.layers.%d.*` | `tensor_naming=2` | All GQA (0 SSM) |

### Dynamic Dimensions (from GGUF)

| Field | Qwen3.6 | DiffusionGemma | Gemma 4 |
|-------|---------|----------------|---------|
| d_model | 2048 | 2816 | 3840 |
| head_dim | 256 | 256/512 (LARGE) | 256 |
| q_heads | 16 | 16 | 16 |
| kv_heads | 2 (GQA layers) | 8 (normal) / 2 (LARGE) | 8 |
| n_experts | 256 | 128 | 0 |
| n_active | 12 | 8 | N/A |
| d_ff | 1024 | 704 | 15360 |

---

## Model Component Fate

| Component | Fate | Notes |
|-----------|------|-------|
| `wubu_model.c` | 🔄 Evolved | Multi-model adapter with `g_tensor_naming` global |
| `wubu_ssm.c` | 🔄 Dual-use | SSM functions (Qwen-only) + GQA functions (all models, with `d_model` param) |
| `wubu_moe.c` | 🔄 Shared | MoE forward used by Qwen and DiffusionGemma |
| `gguf_reader.c` | ✅ Shared | Architecture-agnostic |
| KV cache (Q4_0) | ✅ Shared | Dynamic sizing per-model |
| `wubu_gemma4.c` | 🆕 Dedicated | Gemma 4-specific forward (separate engine) |
| CUDA kernels | 🔄 Per-model | Need ISWA kernels for Gemma, GQA kernels for DiffusionGemma |

---

## Blockers

### P0 — DiffusionGemma Forward Crash

**Symptom**: Model loads successfully (30 GQA layers), crashes during decode with "tensor too large (512 elems, max 256)" for LARGE layers.

**Root cause**: GQA loading code uses fixed `attn_q_norm` weight size of `GQA_HEAD_DIM=256`, but DiffusionGemma LARGE layers have `head_dim=512` (q_norm weight has 512 elements).

**Fix needed**: Per-layer weight buffer sizing from GGUF tensor shapes instead of hardcoded macros. The `gqa_layer_weights` struct already has `head_dim` and `is_large` fields — need to extract these from GGUF during init and use them for buffer allocation in the load path.

### P1 — Gemma 4 12B Benchmark

Dedicated engine in `wubu_gemma4.c` with separate model loading/forward. Hasn't been integrated into the main `bench_512k_full` path yet.

### P2 — GPU Forward (All Models)

CPU-only path works. GPU kernels exist but aren't fully wired for any model's complete forward pass.

---

## Next Steps (Priority Order)

1. 🔧 **Fix DGemma LARGE layer head_dim** — extract per-layer dims for weight buffer allocation
2. 🔧 **Complete DGemma forward** — verify decode produces output
3. 🧪 **Benchmark all 3 models** — `./bench_512k_full` for each at 4K context
4. 📊 **Compare results** — tokens/s, VRAM, accuracy per model
5. 🚀 **GPU forward** — wire kernels for at least one model end-to-end

---

## Key Paths

- Source: `/home/wubu/bytropix/`
- DiffusionGemma: `/home/wubu/models/DiffusionGemma-26B-Q4_K_M.gguf`
- Gemma 4 12B: `/home/wubu/models/gemma4/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`
- Qwen3.6: `/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf`
- Mind palace: `.hermes/mind-palace/paradigm-shift-gemma4.md`
- DGemma notes: `.hermes/mind-palace/tier4-validation/13-benchmarks/gemma4-baseline.md`
