# Token-Superposition Training (TST)

**Paper:** [Efficient Pre-Training with Token Superposition](https://arxiv.org/abs/2605.06546)
**Authors:** Bowen Peng, Théo Gigant, Jeffrey Quesnelle (Nous Research)
**Date:** 7 May 2026
**PDF:** `references/2605.06546_token_superposition.pdf`

## Core Idea

Drop-in method that speeds up pre-training **up to 2.5x** without modifying parallelism, optimizer, tokenizer, data, or model architecture.

### Two Phases

1. **Superposition Phase** (ratio `r` of total steps):
   - Group `s` contiguous tokens into non-overlapping "bags"
   - Average the embeddings of each bag → one "s-token" per bag
   - Sequence length shrinks: `L' = L / s`
   - **To keep equal-FLOPs**: multiply sequence length by `s` (same tokens/step)
   - Loss = Multi-Hot Cross-Entropy (MCE): average of s standard CE losses over each target in the next bag
   - Optimizer: AdamW (β1=0.9, β2=0.95), LR sweep-optimized

2. **Recovery Phase** (ratio `1-r` of total steps):
   - Remove all TST code
   - Standard next-token prediction CE loss
   - Model weights carry over from superposition phase

### Key Hyperparameters

| Param | Best Value | Notes |
|-------|-----------|-------|
| Bag size `s` | 6–8 (dense), 16 (MoE 10B) | Larger s = more throughput, diminishing returns past 8 |
| Step ratio `r` | 0.3 (dense), 0.25 (MoE 10B) | ~25-30% of steps in superposition phase |
| Model sizes tested | 270M, 600M, 3B, 10B A1B MoE | Works across scales |
| **Speedup (equal-loss)** | **2.5x on 10B A1B** | 4768 vs 12311 B200-hours |

### Implementation (from Appendix A)

**Input (embedding superposition):**
```
if tokens.shape == (B, L/s, s):  # superposition mode
    h = tok_embeddings(tokens[..., 0]).float()
    for i in range(1, s):
        h += tok_embeddings(tokens[..., i]).float()
    h = (h / s).to(original_dtype)
```

**Loss (MCE = avg of s CE losses):**
```
# pred shape: (B, L/s, V), labels shape: (B, L)
superposition_bag_size = label_seq // seq_len
offset = superposition_bag_size - 1

# Pad labels with -100 (ignore), shift left by offset
labels = pad(labels, (0, offset), value=-100)[..., offset:]
labels = labels.view(B, seq_len, superposition_bag_size)

# Average CE over s targets
loss = 0
for i in range(superposition_bag_size):
    target = labels[..., i].flatten(0, 1)
    loss += cross_entropy(pred, target)
loss /= s
```

### Relevance to WuBuText AI

TST is **directly applicable** to Phase 3 (training loop):
- Our model (Qwen3.6-35B-A3B) is a MoE similar to the 10B A1B tested
- TST requires **no architecture changes** — just modify the embedding lookup and loss function
- During superposition phase: bag `s` tokens → average embeddings → forward pass on `L/s` tokens → MCE loss on next bag
- During recovery phase: standard CE with original token granularity
- **Tokenizing with our BBPE** still works — TST specifically keeps the tokenizer unchanged
- Target: 2x+ speedup on our pre-training

### Results (10B A1B MoE)

| Metric | Baseline | TST |
|--------|----------|-----|
| Total tokens seen | 1.05T | 2T |
| B200-hours | 12,311 | 4,768 |
| Final loss | 2.252 | 2.236 |
| HellaSwag | 70.1 | 71.2 |
| ARC-C | 46.3 | 47.3 |
| MMLU | 37.4 | 39.0 |
| **Speedup (equal-loss)** | 1x | **2.58x** |
