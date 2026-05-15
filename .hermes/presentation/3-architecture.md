# 3. WuBuText AI — System Architecture

## Overview

WuBuText AI is a **40-layer hybrid** model: 30× SSM (Gated Delta Net) + 10× GQA full attention, initialized from Qwen3.6-35B-A3B GGUF weights. The architecture maps Euclidean representations into the Poincaré ball for hyperbolic computation and uses Token-Superposition Training (TST) for efficient pre-training. All code in pure C + CUDA, no framework dependencies.

---

## 1. GGUF Reader (`include/gguf_reader.h`, `src/gguf_reader.c`)

Parses GGUF format files (Qwen3.6-35B-A3B) — header metadata, tensor info table, and raw weight data. Supports GGML quantized types (Q4_0–Q8_K, IQ2_XS, IQ1_S). Outputs tensor names, shapes, types, and data offsets for downstream loading. Provides `gguf_tensor_info` struct and functions for opening, reading, and closing GGUF files.

---

## 2. Poincaré Embedding Mapping (`include/wubu_mobius.h`, `src/wubu_mobius.c`)

Maps Euclidean token embeddings into the Poincaré ball of radius R (curvature = -1/R²) via the exponential map:

- **exp_map(v)**: Projects a tangent vector v onto the ball
- **log_map(x)**: Inverse — maps a ball point back to tangent space
- **Poincaré linear combination**: Sums points in tangent space then projects back — more stable than chaining Möbius additions

All token embeddings from the GGUF reader pass through `exp_map` on entry and `log_map` on exit (for the LM head projection).

---

## 3. SSM Gated Delta Net — 30/40 layers (`include/wubu_ssm.h`, `src/wubu_ssm.c`)

Mamba2-style structured SSM with Gated Delta Net architecture from Qwen3.6:

- **Conv1d** (kernel=4) — causal depthwise convolution on fused Q/K/V tokens
- **Gated Delta Net recurrence** — per-head recurrent state update with learnable alpha (forgetting), beta (input gate), and dt (discretization timestep)
- **Gated normalization** — RMSNorm per head × SiLU gate
- **Output projection** — VALUE_DIM → D_MODEL

**Dimensions**: D_MODEL=2048, D_INNER=4096, SSM_K_HEADS=16, SSM_V_HEADS=32, SSM_D_STATE=128, DT_RANK=32.

The 30 SSM layers form the "efficient" backbone — linear-time in sequence length, no attention matrix.

---

## 4. GQA Full Attention — 10/40 layers (`include/wubu_ssm.h`, `src/wubu_ssm.c`)

Grouped-Query Attention (GQA) with MRoPE (Multi-head RoPE from Qwen3.6):

- **Q heads / KV heads**: 16 / 2 (8:1 GQA ratio)
- **Head dimension**: 256
- **RMSNorm** applied to Q and K before dot product
- **Gate mechanism**: Fused Q+gate projection, gate applied after causal softmax-weighted V sum
- **Causal mask**: Standard autoregressive lower-triangular attention

The 10 attention layers are interleaved among the SSM layers according to `layer_types[]` (config-defined per Qwen3.6 architecture). These provide full quadratic attention where the model needs it most.

---

## 5. Möbius Gyration (`include/wubu_mobius.h`)

Hyperbolic gyrovector operations enabling non-Euclidean transformations within the Poincaré ball:

- **Möbius addition** (x ⊕ y) — gyrovector group addition, numerically stable formula per Ganea et al. (2018)
- **Möbius scalar multiplication** (r ⊗ x) — via exp_map(r · log_map(x))
- **Möbius gyration** gyr[x,y]z — captures non-associativity of Möbius addition; used for weight-direction adjustments during hyperbolic training
- **Poincaré geodesic distance** — d(x,y) = R · artanh(||(-x) ⊕ y|| / R)
- **Conformal (Lorentz) factor** λ_x = 2R²/(R² − ||x||²)

Phase 2.2 uses tangent-space linear combinations (avoids Möbius add chains for stability); full gyration is reserved for Phase 3 training.

---

## 6. CUDA Kernel Layer (`include/cuda_kernels.h`, `src/cuda_kernels.cu`)

GPU acceleration for all compute-heavy operations:

- **cuBLAS matmul** (`cublasSgemm`) — covers ~90% of FLOPs
- **Element-wise ops** — SiLU, sigmoid, softplus, exp
- **RMSNorm & L2 norm** — fused per-batch kernels
- **Causal conv1d** — depthwise 1D convolution (kernel=4)
- **Gated Delta Net step** — single-head, single-token recurrence kernel
- **GQA forward** — fused Q/K RMSNorm + causal dot-product attention + softmax + V-weighted sum + gate, all in one kernel with scratch buffer
- **Memory management** — device alloc/free, host↔device transfer

**Compiled with**: `nvcc -arch=sm_120` (Blackwell architecture), linked against `-lcublas -lcudart`.

---

## 7. TST Training Loop (`train_integrated`, ✅ integrated)

Token-Superposition Training (TST, arXiv:2605.06546) for up to 2.5× pre-training speedup:

- **Superposition phase** (25% of steps): Average bag-of-s embeddings (s=8), compute Multi-Hot Cross-Entropy loss, processes s× tokens at same FLOPs
- **Recovery phase** (75% of steps): Standard next-token CE loss, weights carry over from superposition
- **Dual optimizer**: AdamW for Euclidean params (output weight, norms, biases); Riemannian SGD for Poincaré ball params (embeddings after exp_map)
- **Checkpointing**: GGUF-compatible tensor format every 1000 steps
- **Memory strategy**: Q5_K/Q8_K weights → f16 on-the-fly dequant, optimizer states offloaded to CPU

Implemented as a standalone C training loop — no PyTorch, no JAX, no external framework.

---

## Data Flow

```
GGUF File  ──→  GGUF Reader  ──→  Token Embeddings (Euclidean)
                                        ↓ exp_map
                                 Poincaré Ball
                                    ↓
                   ┌──────────────────────────────┐
                   │  30× SSM Gated Delta Net      │  (linear-time)
                   │  +                             │
                   │  10× GQA Full Attention        │  (quadratic, sparse)
                   └──────────────────────────────┘
                                    ↓ log_map
                                 Euclidean LM Head
                                    ↓
                              TST Loss (MCE / CE)
```

See `../../DIAGRAMS/wubu-math-pipeline.svg` for the Poincaré↔Euclidean pipeline and `../../DIAGRAMS/gguf-rip-pipeline.svg` for the GGUF extraction flow.
