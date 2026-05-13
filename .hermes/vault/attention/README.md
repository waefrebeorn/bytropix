# Vault: Attention — 4 Variants
#
## 1. WuBuSparseAttention (`wubu-sparse-attention/`)
*Best for C/CUDA port — clean PyTorch, standard ops*
- Dual memory: Working (recent 256 dense) + Associative (historic sparse index)
- RAS indexer: Q/K → 64-dim → top-k (64) → gather full K/V → sparse attention
- 6-layer decoder, d_model=512, 8 heads
- Complexity: O(n·k) + O(n·W) ≈ O(n·320) — effectively linear

## 2. Topological Sequence Model (`topological-sequence-model/`)
*Inspired the Hamilton encoder CUDA kernel — O(n) complexity*
- Conv1D (stride=4) compression → Physics state (δ, χ, radius) → Transposed Conv1D
- `poincare_simulation(δ, χ) = cos(δ/2) + i·sin(δ/2)·sin(2χ)` — complex valued
- 13.6M params (vs GPT-2 117M), 4× less memory
- JAX/Flax, Q-Learner LR, multi-GPU pmap
- Port viability: high — Conv1D/transposed are well-supported in CUDA

## 3. Hyperbolic Attention (`hyperbolic-attention/`)
*The "Tri-Cameral Mind" — pedagogical, not a real attention layer*
- Body (weights) + Soul (integer topology on torus) + Echo (residual float)
- Gradient decomposition: amplify ×50-100, decompose into quotient/remainder
- 5 evolutionary variants (Perceptron → Fixed → Stabilized → Final + TgT Test)
- Port viability: low — 2D perceptron toy, not transformer attention

## 4. Entropix Sampler (`entropix-sampler/`)
*Inference-time dynamic sampling — not attention*
- DSState: 12-field NamedTuple tracking entropy/varentropy/temperature/Dirichlet
LV/HELV/LEHV/HEHV → accept/explore/resample
    29|- `fit_dirichlet()` — Halley's method, heavy jax.scipy.special dependency
    30|- Port viability: low — gamma/polygamma not natively in CUDA
    31|

---

*Part of the WuBuText AI project. See [Project Overview](../../README.md) and [Presentation Layer](../presentation/README.md) for navigation.*
