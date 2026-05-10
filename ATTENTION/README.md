# Attention: Beyond Dot-Products

The WuBu project didn't stop at quadratic attention. This folder collects experiments in alternative attention mechanisms — sparse associative memory, hyperbolic kNN attention, entropic sampling, and topological sequence models.

## Contents

### wubu-sparse-attention/
**`WuBuSparseAttention.py`** — A dual-memory attention mechanism:
- **Working Memory**: dense cache of recent tokens (standard attention)
- **Associative Memory**: sparse store of historical tokens, indexed by a lightweight RAS (Retrieval-Augmented Sparse) indexer
- The indexer uses low-rank projections (d_model → 64) for fast relevance scoring, then gathers top-k from full-resolution K/V
- Effectively O(n·k) instead of O(n²) for long sequences

### hyperbolic-attention/
**`Wubu_Clockwork_Final.py`**, **`Wubu_Clockwork_Stabilized.py`**, **`Wubu_Clockwork_Perceptron.py`**, **`Wubu_Clockwork_Fixed.py`**
- The "Clockwork" family: attention with temporal gating and periodic activation
- Each head operates at a different frequency — inspired by clock neurons
- Mixes hyperbolic attention (kNN in Poincaré ball) with standard attention

**`WuBu_TgT_Test.py`** — Token-to-Grid Token test: reshapes token sequences into 2D grids for spatial conv-attention hybrids.

### entropix-sampler/
**`xjdr_backup_sampler.py`** — An adaptation of the Entropix dynamic sampler for image tokenization. Uses:
- Dirichlet support scores for token diversity
- Entropy/Varentropy tracking for adaptive temperature
- Outlier detection via top-k entropy thresholds
- Purely JAX, registered as a proper PyTree

### topological-sequence-model/
**`topological_sequence_model1.py`**, **`bullshitdeepseekTPM.md`**
- A linear-complexity attention mechanism using topological compression
- Encodes sequences into Hamiltonian parameters (δ, χ, radius) via convolution
- Simulates attention through Poincaré geometry: `poincare(δ, χ) = cos(δ/2) + i·sin(δ/2)·sin(2χ)`
- Decompresses via transposed convolution back to sequence length
- **The math that inspired the Hamilton encoder CUDA kernel**

## The Story

The attention arc shows WuBu's progression:
1. **Standard attention** (WuBuMindV1)
2. **Sparse working/associative memory** (WuBuSparseAttention)
3. **Hyperbolic kNN attention** (kNNHyperbolicAttentionLayer in WuBuMindJAX)
4. **Topological compression** (TSM — the bullshitdeepseek paper)
5. **Entropic dynamic sampling** (Entropix adaptation)
6. **CUDA Hamilton encoder** (LLAMA-CPP-INTEGRATION — the production version)
