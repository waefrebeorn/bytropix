# bytropix — WuBu Text AI

**Pure C + CUDA: Qwen3.6-35B-A3B with WuBu nested hyperbolic geometry.**
*Inference debugging active — all dequants verified, SSM divergence remaining.*

---

## Current State (May 17 PM — Corrected)

**Inference runs but produces WRONG output:**
- Us: `<|endoftext|>Hello_vendor` (prefill top-token: 'ore' @ 9.89)
- Ref (llama.cpp): `Hello Here's a thinking process:`
- **Root cause: SSM L0 cos_sim=0.40 vs llama.cpp** (before MoE runs)

### What's Verified Correct

| Dequant | Status | Used In | Verified Against |
|---------|--------|---------|-----------------|
| IQ2_XXS (16) | ✅ | gate_exps, up_exps (all layers) | llama.cpp dequantize_row_iq2_xxs() |
| IQ3_XXS (18) | ✅ | down_exps (37/40 layers) | llama.cpp dequantize_row_iq3_xxs() |
| IQ4_XS (23) | ℹ️ Untested | down_exps (L34, 38, 39) | Written from ref, needs cross-dump |
| IQ2_S (22) | ✅ | — (not in this GGUF) | llama.cpp dequantize_row_iq2_s() |
| IQ1_M (29) | ✅ Fixed | — (not in this GGUF) | Was broken (-1.0f delta, wrong scale index) |
| Q5_K (13) | ✅ | attn_gate, qkv, shexp, embd | llama.cpp dequantize_row_q5_K() |
| Q6_K (14) | ✅ | ssm_out, shexp down | llama.cpp dequantize_row_q6_K() |
| Q4_K (12) | ✅ | output.weight | llama.cpp dequantize_row_q4_K() |

### Actual Tensor Types (from GGUF — labels corrected per DA v9)

| Tensor | Type | Details |
|--------|------|---------|
| ffn_down_exps | **IQ3_XXS** (type 18) | 37/40 layers (0-33, 35-37) |
| ffn_down_exps | **IQ4_XS** (type 23) | 3/40 layers (34, 38, 39) |
| ffn_gate_exps | **IQ2_XXS** (type 16) | All layers |
| ffn_up_exps | **IQ2_XXS** (type 16) | All layers |
| ffn_gate_inp | **F32** | Router weights |
| shexp gate/up | **Q5_K** | Shared expert |
| shexp down | **Q6_K** | Shared expert |
| ssm_out | **Q6_K** | SSM output projection |
| output.weight | **Q4_K** | Language model head |
| token_embd | **Q5_K** | Input embeddings |

> ⚠️ **DA v9 Fix:** Python `tools/dump_gguf.py` had WRONG type labels — type 18 mapped to "IQ2_S" (actually IQ3_XXS), type 23 to "IQ1_M" (actually IQ4_XS). Now corrected.

---

## Pipeline Overview

![Inference Pipeline](DIAGRAMS/inference-pipeline.svg)

The current inference flow: **Tokenize** (GGUF native) → **Embed** (Q5_K dequant) → **40 Layers** (30 SSM Gated DeltaNet + 10 GQA full attention, each with RMSNorm → SSM/GQA → MoE router → top-8 experts + shared expert → residual) → **Output Projection** (cublasSgemm 248K×2048) → **Sampling**.

![Quant Layer Map](DIAGRAMS/quant-layer-map.svg)

![Bug Status](DIAGRAMS/bug-status.svg)

---

## Quick Start

```bash
# Build
make infer_text

# Run inference (CPU, NOGPU=1)
NOGPU=1 ./infer_text /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "Hello" 12

# Run inference (GPU)
./infer_text /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "Hello" 12

# Reference comparison
cd ~/llama.cpp/build/bin
./llama-cli -m /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -p "Hello" -n 5 --temp 0.0
```

**CUDA:** `/usr/local/cuda-13.1/bin/nvcc -arch=sm_120` | **GPU:** RTX 5050 6.4GB | **Model:** Qwen3.6-35B-A3B-UD-IQ2_M.gguf

---

## Architecture

### Model Spec (Qwen3.6-35B-A3B)

| Component | Value |
|-----------|-------|
| Layers | 40 (30 SSM Gated DeltaNet + 10 GQA full attention) |
| Hidden dim | 2048 |
| Context | 262K native |
| Vocab | 248,320 |
| MoE | 256 experts, 8 active + 1 shared per token |
| SSM heads | 16 K-heads × 128, 32 V-heads × 128 |
| GQA heads | 16 Q × 256, 2 KV × 256 |
| Expert FFN dim | 512 |
| RoPE | θ=10,000,000, 0.25 partial rotary factor, MRoPE 3D sections=[11,11,10,0] |
| Quant | Mixed: IQ2_XXS / IQ3_XXS / IQ4_XS / Q5_K / Q6_K / Q4_K |

### Layer Structure

Each of the 40 layers follows:
1. **RMS Norm** — `rms_norm(x, weight, eps=1e-6)`
2. **SSM (30 layers) or GQA (10 layers)** — alternating every 4 layers
3. **MoE** — router (F32 gate_inp × 256 experts) → top-8 active experts (IQ2_XXS gate/up + IQ3_XXS/IQ4_XS down) + shared expert (Q5_K/Q6_K) → SiLU-gated weighted sum
4. **Residual** — `x += moe_output`

### WuBu Hyperbolic Pipeline

```
Euclidean embds → Poincaré ball (R=0.956) → SSM/GQA → MoE → output
                      ↓                         ↓
             95% NN preserved          Nested hyperbolic router
```

Hyperbolic operations implemented in pure C/CUDA:
- exp_map / log_map (Euclidean ↔ Poincaré ball)
- Möbius addition (SSM recurrence)
- Gyration closed-form (3× faster than iterative)
- Poincaré distance (MoE router)
- RSGD optimizer (Riemannian SGD in tangent space)

---

## Phase Roadmap

![Phase Roadmap](DIAGRAMS/phase-roadmap.svg)

| Phase | Component | Status | Notes |
|-------|-----------|--------|-------|
| **0-2** | GGUF + Embed + Attention | ✅ Complete | All 733 tensors, 13 types |
| **3-4** | MoE Port | ✅ Complete | All dequants fixed, lazy dispatch |
| **5-6** | Vision + CUDA | ✅ Complete | GPU ViT, SSM scan kernels |
| **P0** | Inference Debug | 🔴 Active | **SSM L0 cos_sim=0.40 — not a dequant issue** |
| **P1** | IQ4_XS Verify | 🟡 Pending | Needs cross-dump vs llama.cpp |
| **P2** | Spec Decode + API | ⬜ Future | MTP head, OpenAI-compatible server |

---

## Project Structure

```
bytropix/
├── src/              # Core C implementation
│   ├── wubu_ssm.c            SSM Gated Delta Net
│   ├── wubu_mobius.c         Hyperbolic operations
│   ├── wubu_moe.c            MoE forward/backward
│   ├── wubu_poincare_gqa.c   Poincaré attention
│   ├── wubu_nested_ssm.c     Nested hyperbolic SSM
│   ├── gguf_reader.c         GGUF format + dequant
│   ├── cuda_kernels.cu       GPU kernels
│   └── wubu_vision.c         Vision transformer
├── include/          # Headers
├── tools/            # Binaries
│   ├── infer_text.c          CPU inference (debug target)
│   ├── infer_text_gpu.cu     GPU inference
│   └── test_*.c              Unit tests
├── DIAGRAMS/         # 13 SVG architecture diagrams (May 17: +3 updated)
├── .hermes/          # Mind palace + research vault
│   ├── mind-palace/          Prestige system
│   ├── vault/                Research vault
│   └── plans/                DA audits (v1-v9)
└── MATH/lean/        # Formal proofs
```

---

## Key Binaries

| Binary | What It Does | Status |
|--------|-------------|--------|
| `infer_text` | CPU inference (primary debug target) | 🔴 SSM bug |
| `infer_text_gpu` | GPU inference | 🔴 Same SSM bug |
| `test_ssm` | SSM forward verify | 🟢 Verified |
| `test_moe` | MoE forward verify | 🟢 Verified |
| `test_kv_cache` | KV cache correctness | 🟢 max_diff=0 |
| `llama-cli` | Reference (external) | 🟢 Ground truth |

---

## References

- `.hermes/mind-palace/goal-mantra.md` — Session prestige paste
- `.hermes/mind-palace/plans/devils_advocate_v9.md` — Quant layer audit
- `~/llama.cpp/build/bin/llama-cli` — Ground truth reference
- `THEORY/WuBu_Nesting.md` — Original nesting paper
- `MATH/lean/wubu_proofs/` — 4 formal Lean proofs
