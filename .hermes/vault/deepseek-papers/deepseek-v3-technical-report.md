# DeepSeek-V3 Technical Report

**Title**: DeepSeek-V3 Technical Report
**Authors**: DeepSeek-AI (Aixin Liu, Bei Feng, et al.)
**Date**: 2024-12-27
**ArXiv**: https://arxiv.org/abs/2412.19437
**PDF**: https://arxiv.org/pdf/2412.19437

## Abstract

We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages. DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training.

## Key Architectural Findings

### 1. Multi-head Latent Attention (MLA) — CRITICAL
- **What**: Projects keys and values into a low-dimensional latent space before KV cache storage
- **KV cache reduction**: ~75% reduction compared to standard MHA
- **How**: For head dimension d, latent dimension d' where d' << d (typically d' = d/4)
- **Mathematical form**:
  ```
  k_i = W_UK * (W_DK * h_i)     // down-projection then up-projection
  v_i = W_UV * (W_DV * h_i)     // same for values
  Cache: store only the latent vectors (d' dimensions), not full K,V
  ```
- **Relevance to WuBuText**: MLA is an alternative to GQA for KV cache reduction. Our current approach uses GQA (which groups query heads to share KV heads). MLA could potentially reduce KV cache further. Consider combining MLA with our GQA layers for even more efficient 256K context.

### 2. DeepSeekMoE Architecture
- Fine-grained expert segmentation: mN total experts, mK active per token (m > 1)
- Shared experts: Ks experts that are always activated, capturing common knowledge
- Routed experts: the remaining N - Ks experts, with top-K routing
- Normalized sigmoid gating (from DeepSeekMoE paper)
- **Relevance to WuBuText**: This is exactly the scheme used for our 256 experts with 8 active. Implement normalized sigmoid gating in `moe.c`.

### 3. Auxiliary-Loss-Free Load Balancing
- **Problem**: MoE training tends to route most tokens to a few experts (collapse)
- **Standard solution**: Add auxiliary loss to encourage balanced routing
- **DeepSeek-V3 innovation**: Dynamic bias adjustment instead
  - Track expert load; if expert i is overloaded, subtract bias b_i from its gating score
  - If expert i is underloaded, add bias b_i
  - No additional loss term needed
- **Equation**:
  ```
  g'_i = sigmoid(s_i + b_i)
  b_i ← b_i - α * (load_i - target_load)
  ```
- **Relevance to WuBuText**: Implement this in our MoE router. Simpler than auxiliary loss and doesn't interfere with main training objective.

### 4. Multi-Token Prediction (MTP) — CRITICAL for Speculative Decoding
- **What**: Predict D future tokens at each position, not just the next token
- **Architecture**: D independent output heads, each predicting the token at position t+d
- **Training loss**: Sum of cross-entropy losses for all D predictions
- **Benefits**:
  - Stronger representation learning (model must plan ahead)
  - Naturally enables speculative decoding: the model's own D-head predictions serve as draft proposals
  - **Relevance to WuBuText**: MTP is directly applicable to our speculative decoding implementation. Instead of a separate draft model, our model can self-draft using MTP heads. This reduces the "draft model overhead" in our C implementation.

### 5. Training Efficiency
- 2.788M H800 GPU hours for full training (extremely efficient)
- FP8 mixed precision training
- DualPipe algorithm for computation-communication overlap
- No irrecoverable loss spikes during entire training process

### 6. Model Configuration
- n_layers: 70 (for 671B model)
- d_model: 7168
- n_heads: 128 (for MLA)
- n_experts: 256 (routed) + 1 (shared)
- n_active: 8
- **Relevance**: Note: DeepSeek-V3 uses 256 routed experts with 8 active — **exactly the same expert configuration as WuBuText's 256 experts / 8 active!**

## Relevance to WuBuText AI — HIGHLY RELEVANT

| Aspect | DeepSeek-V3 | WuBuText |
|--------|-------------|----------|
| Model Size | 671B total, 37B active | Smaller target but same MoE pattern |
| Experts | 256 routed + 1 shared, 8 active | 256 experts, 8 active (SAME!) |
| Attention | MLA (latent KV compression) | GQA (grouped query attention) |
| Load Balancing | Auxiliary-loss-free bias | Should implement directly |
| MTP | D future tokens predicted | Enables self-speculative decoding |
| Training | 14.8T tokens, 2.788M GPU hrs | Reference for efficiency |

## Key Implementation Notes for WuBuText

### Normalized Sigmoid Gating (C implementation):
```c
// Router: compute logits s_i for each expert
for (int i = 0; i < NUM_EXPERTS; i++) {
    s[i] = dot_product(hidden_state, expert_router_weights[i]);
    g[i] = sigmoid(s[i] + bias[i]);  // bias from load balancing
}
// Top-K selection
top_k_indices = argmax_k(g, K_ACTIVE);
// Normalize
float sum_g = 0;
for (int k = 0; k < K_ACTIVE; k++) sum_g += g[top_k_indices[k]];
for (int k = 0; k < K_ACTIVE; k++) g_normalized[k] = g[top_k_indices[k]] / sum_g;
```

### MTP for Speculative Decoding:
```c
// Self-drafting with MTP heads
for (int d = 1; d <= D; d++) {
    draft_tokens[d] = mtp_head[d](hidden_state);
}
// Verify drafts with main model
// Accept/reject per standard speculative decoding
```

## References
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- DeepSeek-V2 (MLA): https://arxiv.org/abs/2405.04434
- DeepSeekMoE: https://arxiv.org/abs/2401.06066
