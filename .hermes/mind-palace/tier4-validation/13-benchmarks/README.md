# Benchmarks — Validation Framework

## minGPT Functional Comparison — INVALID / DROPPED

**Critical correction:** minGPT is a Python toy model with 100K params on character-level data.
Our model has 3B active params with a 248K vocabulary. These are NOT comparable.

The minGPT comparison was based on the old baseline C code (768 dim, 97-token vocab).
With the Qwen3.6 target (2048 dim, 248K vocab), minGPT is irrelevant as a baseline.

**Replace with:** Comparison against Qwen3.6-base model (same architecture, different initial weights).

## Target Benchmarks

Use these instead:

| Benchmark | What It Measures | Target |
|-----------|-----------------|--------|
| **WikiText-2 perplexity** | Language modeling quality | < 15 (Qwen3.6 base ref needed) |
| **Qwen3.6-base forward pass match** | Weight extraction correctness | Output logits match within Q5_K error |
| **Loss convergence curve** | Training quality | Must decrease monotonically, reach < 5.0 |
| **Expert utilization entropy** | MoE routing quality | > 0.7 × log(256) for uniform routing |
| **Embedding norm stability** | Hyperbolic training safety | All norms < 0.99 during training |
| **Inference speed** | CUDA kernel performance | > 100 tok/s (target: 500+) |
| **VRAM usage** | Memory efficiency | < 6GB peak |

## Training Convergence Metrics

- **Loss curve** vs tokens seen (log-log plot: should be linear)
- **Gradient norm** distribution — track mean/median/max per step
- **Expert utilization** — entropy of expert assignment distribution
  - Target: uniform across 256 experts (entropy = log(256) = 5.55 nats)
  - Warning: if entropy < 2.0, routing collapse
- **Hyperbolic norm distribution** — are Poincaré embeddings staying within ball?
  - Track mean/median/max of ||h|| after each layer
  - Alert if any exceeds 0.99
- **Per-layer DeltaNet vs GQA output difference** — measure ||y_linear - y_softmax|| for same input
  - If very large, the two attention types are learning different functions (expected)
  - If very small, one attention type is redundant

## Performance Benchmarks

| Configuration | Expected Tok/s | Notes |
|---------------|---------------|-------|
| CPU, no CUDA | ~2 tok/s | For testing only |
| CUDA, no optimizations | ~50 tok/s | Straight matmul |
| CUDA + FlashAttention | ~100 tok/s | GQA layers only |
| CUDA + SSM scan kernel | ~300 tok/s | DeltaNet layers |
| CUDA + MoE grouping | ~500 tok/s | All optimizations |

VRAM at each config:
- Batch size 2, seq 4096: ~3.5GB
- Batch size 4, seq 4096: ~4.5GB
- Batch size 2, seq 32768: ~5.5GB
- Batch size 1, seq 262144: ~6GB (KV cache dominates)
