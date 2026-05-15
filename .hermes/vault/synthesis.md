# WuBuText AI — Architectural Synthesis & Prioritized Recommendations

> **Generated**: 2026-05-15
> **Source Papers**: Qwen3, Qwen2.5-1M, DeepSeek-V3, DeepSeek-V3.2, DeepSeekMoE, DeepSeek-R1, Gemma 3, DeepSeek Blog Posts
> **Target**: WuBuText AI — 256K context, 30 SSM + 10 GQA layers, 256 experts (8 active), C inference engine, speculative decoding

---

## Executive Summary

WuBuText's architectural foundation — 30 SSM + 10 GQA layers, 256 experts with 8 active, targeting 256K context — is **independently validated** by multiple frontier labs. Key confirmations:

| Validation | Source | Detail |
|:-----------|:-------|:-------|
| **256 experts / 8 active** | DeepSeek-V3 | Identical expert config (256 routed, 8 active) |
| **SSM-as-local / GQA-as-global** | Gemma 3 | "More local than global" attention ratio — same 3:1 pattern |
| **Hybrid MoE + attention** | Qwen3 | Both dense and MoE variants, our SSM+GQA hybrid is novel |
| **Long context via sparsity** | DeepSeek-V3.2, Qwen2.5-1M | DSA (O(L log L)) and chunked prefill for 256K |

---

## PART 1: Complete Inventory of Architectural Innovations

### A. MoE Architecture Innovations

| # | Innovation | Source Paper | Key Detail | C Impl Required? |
|---|-----------|-------------|-----------|------------------|
| A1 | **Fine-grained expert segmentation** | DeepSeekMoE | Use mN experts, activate mK — more experts = finer specialization. Our 256/8 already qualifies. | No (already designed) |
| A2 | **Shared experts** | DeepSeekMoE | Ks experts always active, capturing common knowledge. Reduces redundancy among routed experts. | **YES** — `moe.c` |
| A3 | **Normalized sigmoid gating** | DeepSeekMoE, DeepSeek-V3 | `g_i = sigmoid(s_i) / sum(sigmoid(s_j))` — independent per-expert activation, avoids softmax winner-take-all. | **YES** — `moe.c` |
| A4 | **Auxiliary-loss-free load balancing** | DeepSeek-V3 | Dynamic bias adjustment `b_i ← b_i - α*(load_i - target_load)` instead of auxiliary loss. Simpler, no interference with training. | **YES** — `moe.c` |
| A5 | **Top-K routing with shared expert isolation** | DeepSeekMoE, DeepSeek-V3 | Select top-K from routed experts only; shared experts always included. Normalize over combined set. | **YES** — `moe.c` |

### B. Attention / Context Innovations

| # | Innovation | Source Paper | Key Detail | C Impl Required? |
|---|-----------|-------------|-----------|------------------|
| B1 | **Multi-head Latent Attention (MLA)** | DeepSeek-V3 | Project K/V into latent space (d' = d/4). ~75% KV cache reduction vs MHA. Alternative/supplement to GQA. | **YES** — `attention.c` |
| B2 | **DeepSeek Sparse Attention (DSA)** | DeepSeek-V3.2 | O(L log L) sparse attention: local window + global positions. Combine with SSM for full linear-time inference. | **YES** — `attention.c` |
| B3 | **Chunked prefill** | Qwen2.5-1M | Split long prompts into chunks of size C. 3-7x prefill speedup. Critical for 256K first-token latency. | **YES** — `inference.c` |
| B4 | **RoPE length extrapolation** | Qwen2.5-1M | 4x context extension via frequency scaling `θ_i = base^(-2i/d) × scale_factor` with scale_factor < 1. Train at 64K, extrapolate to 256K. | **YES** — `attention.c` |
| B5 | **Local/Global attention ratio** | Gemma 3 | More local layers than global. SSM = local (linear state), GQA = global. **Validates 30:10 ratio.** | No (already designed) |
| B6 | **Sliding window attention** | Qwen3 | Fixed-size local window for efficiency. SSM already provides this more efficiently. | No (SSM supersedes) |
| B7 | **Progressive context training** | Qwen2.5-1M | 4K → 32K → 128K → 1M staged training. Each stage adds long-range data. | Training pipeline |

### C. Multi-Token & Decoding Innovations

| # | Innovation | Source Paper | Key Detail | C Impl Required? |
|---|-----------|-------------|-----------|------------------|
| C1 | **Multi-Token Prediction (MTP)** | DeepSeek-V3 | D independent output heads predict future tokens. Training loss: sum of D cross-entropy terms. Enables self-speculative decoding. | **YES** — `model.c`, `speculative.c` |
| C2 | **Self-speculative decoding via MTP** | DeepSeek-V3, Blog | No separate draft model needed. Model drafts from its own MTP heads, verifies with main model. Reduces draft model overhead in C impl. | **YES** — `speculative.c` |
| C3 | **Thinking/non-thinking mode** | Qwen3 | Token-controlled mode switch. `max_thinking_tokens` budget parameter. Implement as C runtime flag for inference path selection. | **YES** — `inference.c` |

### D. Training & Post-Training Innovations

| # | Innovation | Source Paper | Key Detail | C Impl Required? |
|---|-----------|-------------|-----------|------------------|
| D1 | **Pure RL for reasoning** | DeepSeek-R1 | No human reasoning traces needed. Emergent: self-reflection, verification, backtracking. | Training pipeline |
| D2 | **Knowledge distillation** | DeepSeek-R1, Gemma 3 | Distill reasoning from large models. Gemma3-4B competitive with Gemma2-27B via distillation. | Training pipeline |
| D3 | **FP8 mixed precision training** | DeepSeek-V3 | Full FP8 training at scale. No irrecoverable loss spikes. | Training pipeline |
| D4 | **DualPipe algorithm** | DeepSeek-V3 | Overlap computation and communication across GPUs. | Training pipeline |
| D5 | **Agentic task synthesis** | DeepSeek-V3.2 | Pipeline for generating agentic training data. | Future |

### E. Framework Validations for WuBuText

| # | Finding | Source | Implication |
|---|--------|--------|-------------|
| E1 | 256 experts / 8 active = **identical to DeepSeek-V3** | DeepSeek-V3 | DeepSeek validates this granularity at 671B scale |
| E2 | 30 SSM : 10 GQA = same pattern as Gemma 3's local:global ratio | Gemma 3 | Major lab validates "more local than global" design |
| E3 | SSM replaces local attention entirely, GQA handles global | Gemma 3 | Our hybrid is more sophisticated than Gemma 3's approach |
| E4 | MTP + self-speculative decoding eliminates draft model overhead | DeepSeek-V3 | Directly applicable to our speculative decoding plans |

---

## PART 2: Priority Map — What to Implement & In What Order

### P0: CRITICAL — Must Implement (Highest Impact on 256K / Inference Speed)

| Priority | Innovation | Why P0 | C File | Effort |
|:--------:|-----------|--------|--------|:------:|
| P0.1 | **Normalized sigmoid gating** | Core correctness — MoE doesn't work well without it. Softmax gating causes routing collapse. | `moe.c` | Low |
| P0.2 | **Auxiliary-loss-free load balancing** | Prevents expert collapse during training. No loss term interference. Dynamic bias adjustment. | `moe.c` | Low |
| P0.3 | **Chunked prefill** | Critical for 256K first-token latency. 3-7x prefill speedup documented by Qwen. Without this, 256K prompt = O(L²) dead. | `inference.c` | Medium |
| P0.4 | **RoPE length extrapolation** | Train at 64K, infer at 256K without retraining. Single frequency scale factor change. | `attention.c` | Low |

**Implementation Order**: Normalized sigmoid gating → Load balancing → Chunked prefill → RoPE extrapolation

### P1: HIGH — Should Implement (Major Quality/Speed Gains)

| Priority | Innovation | Why P1 | C File | Effort |
|:--------:|-----------|--------|--------|:------:|
| P1.1 | **Multi-Token Prediction (MTP)** | Dual benefit: better training signal + enables self-speculative decoding. D output heads. | `model.c`, `speculative.c` | High |
| P1.2 | **Self-speculative decoding via MTP** | Eliminates need for separate draft model. Model self-drafts via D heads, verifies in one pass. | `speculative.c` | Medium |
| P1.3 | **DeepSeek Sparse Attention (DSA)** | O(L log L) for GQA layers at 256K. Combined with SSM's O(L) = full linear-time inference. | `attention.c` | High |
| P1.4 | **Shared experts** | Reduces redundancy, improves specialization. 4-8 shared experts + 248-252 routed. | `moe.c` | Medium |
| P1.5 | **Thinking/non-thinking mode** | Unified model for reasoning vs speed. Runtime flag in C inference engine. | `inference.c` | Low |

### P2: MEDIUM — Implement for Quality (Post-Training / Future)

| Priority | Innovation | Why P2 | When |
|:--------:|-----------|--------|:----:|
| P2.1 | **Multi-head Latent Attention (MLA)** | Alternative KV cache reduction. Consider if GQA alone isn't enough at 256K. Combines with DSA. | After 256K baseline works |
| P2.2 | **Progressive context training** | 4K → 32K → 128K → 256K staged training. | Training pipeline phase |
| P2.3 | **Pure RL for reasoning** | DeepSeek-R1 style. Emergent self-reflection, verification. | Post-training phase |
| P2.4 | **Knowledge distillation** | Distill from larger reasoning model for smaller WuBuText variant. | Post-training phase |
| P2.5 | **FP8 mixed precision** | Training efficiency. DeepSeek-V3 proved no loss spikes at scale. | Training pipeline |

### P3: LOW — Monitor / Future Research

| Innovation | Source | Notes |
|-----------|--------|-------|
| DualPipe algorithm | DeepSeek-V3 | Multi-GPU training optimization |
| Agentic task synthesis | DeepSeek-V3.2 | For agent capabilities, later phase |
| Vision understanding | Gemma 3 | Multimodal extension, separate track |

---

## PART 3: C Implementation Blueprint

### 3.1 Normalized Sigmoid Gating + Shared Experts (`moe.c`)

```c
// DeepSeekMoE forward pass with shared experts & normalized sigmoid gating

// Step 1: Compute router logits
float router_logits[NUM_ROUTED_EXPERTS];
for (int i = 0; i < NUM_ROUTED_EXPERTS; i++) {
    router_logits[i] = dot_product(hidden, router_weight[i]);
}

// Step 2: Sigmoid gating with load-balancing bias
float gate_scores[NUM_ROUTED_EXPERTS];
for (int i = 0; i < NUM_ROUTED_EXPERTS; i++) {
    gate_scores[i] = sigmoid(router_logits[i] + load_bias[i]);
    // load_bias[i] updated during training: b_i -= alpha * (load_i - target_load)
}

// Step 3: Always include shared experts
int selected[N_SHARED + K_ACTIVE];
int sel_count = 0;
for (int i = 0; i < N_SHARED; i++) {
    selected[sel_count++] = SHARED_EXPERT_IDS[i];  // e.g., 0..N_SHARED-1
}

// Step 4: Top-K from routed experts (exclude shared indices)
top_k_indices = argmax_k(gate_scores, K_ACTIVE);
for (int k = 0; k < K_ACTIVE; k++) {
    selected[sel_count++] = top_k_indices[k];
}

// Step 5: Normalize over all selected
float sum_g = 0;
for (int i = 0; i < sel_count; i++) sum_g += gate_scores[selected[i]];
if (sum_g > 1e-10f) {
    for (int i = 0; i < sel_count; i++) gate_scores[selected[i]] /= sum_g;
}

// Step 6: Compute weighted sum of expert outputs
memset(output, 0, d_model * sizeof(float));
for (int i = 0; i < sel_count; i++) {
    int eid = selected[i];
    float weight = gate_scores[eid];
    float* expert_out = expert_forward(eid, hidden);  // FFN(eid, hidden)
    for (int j = 0; j < d_model; j++) {
        output[j] += weight * expert_out[j];
    }
}
```

**Key parameters**: `N_SHARED = 4` to `8`, `NUM_ROUTED = 248` to `252`, `K_ACTIVE = 8`

### 3.2 Auxiliary-Loss-Free Load Balancing Update

```c
// Per-step/Training Step update
// Track load: how many tokens were routed to each expert
float load_i = expert_token_count[i] / total_tokens;
float target_load = K_ACTIVE / (float)NUM_ROUTED_EXPERTS;

// Update bias (moving average style)
float alpha = 0.001f;  // learning rate for bias
load_bias[i] -= alpha * (load_i - target_load);

// Clamp to prevent extreme bias values
if (load_bias[i] > MAX_BIAS) load_bias[i] = MAX_BIAS;
if (load_bias[i] < -MAX_BIAS) load_bias[i] = -MAX_BIAS;
```

### 3.3 Chunked Prefill (`inference.c`)

```c
// For a prompt of length L (potentially 256K tokens)
#define CHUNK_SIZE 4096  // or adaptive based on available memory

void chunked_prefill(float* kv_cache, int* tokens, int L) {
    for (int chunk_start = 0; chunk_start < L; chunk_start += CHUNK_SIZE) {
        int chunk_end = min(chunk_start + CHUNK_SIZE, L);
        int chunk_len = chunk_end - chunk_start;
        
        // Process this chunk through all 40 layers
        // For SSM layers (30): linear time, just update state
        // For GQA layers (10): attend to this chunk + all previous KV
        //   - First chunk: process fully (no previous KV)
        //   - Subsequent chunks: attend to this chunk + KV from prior chunks
        process_chunk(tokens + chunk_start, chunk_len, 
                      chunk_start,  // position offset
                      kv_cache);     // growing KV cache
    }
}
```

### 3.4 RoPE Length Extrapolation (`attention.c`)

```c
// RoPE frequency computation with extrapolation scaling
// Train at max_pos = 65536, infer at max_pos = 262144 (4x extrapolation)

void init_rope_frequencies(float* freqs, int d_model, float base_freq) {
    float scale_factor = 0.25f;  // 4x extrapolation: 64K → 256K
    // From Qwen2.5-1M: scale_factor < 1 extends effective context
    
    for (int i = 0; i < d_model / 2; i++) {
        float theta = powf(base_freq, -2.0f * i / d_model);
        freqs[i] = theta * scale_factor;
    }
}

// At inference: apply to position p where p can be up to 262144
// even though model was trained only up to 65536
float cos_val = cosf(p * freqs[i]);
float sin_val = sinf(p * freqs[i]);
```

### 3.5 Multi-Token Prediction + Self-Speculative Decoding

```c
// MTP Architecture: D output heads
// During training, each head predicts token at position t+d
// During inference, use heads for self-speculative decoding

typedef struct {
    float* weight;  // [d_model, vocab_size] per head
    float* bias;    // [vocab_size] per head
} MTPHead;

MTPHead mtp_heads[D];  // D = 2 to 5 (DeepSeek-V3 uses D=2)

// Self-speculative decoding
void speculative_decode(int* output_tokens, int max_tokens) {
    while (num_generated < max_tokens) {
        // Step 1: Draft D tokens from MTP heads in one forward pass
        int draft_tokens[D];
        for (int d = 0; d < D; d++) {
            float* logits = mtp_forward(mtp_heads[d], hidden_state);
            draft_tokens[d] = argmax(logits);
        }
        
        // Step 2: Verify all drafts with main model forward pass
        // Main model processes all draft tokens, produces logits for each position
        acceptance_probs = verify_drafts(draft_tokens, D);
        
        // Step 3: Accept/reject per standard speculative decoding
        int accepted = sample_acceptance(acceptance_probs);
        for (int i = 0; i < accepted; i++) {
            output_tokens[num_generated++] = draft_tokens[i];
        }
        // If rejection: resample from corrected distribution
    }
}
```

### 3.6 DSA for GQA Layers (`attention.c`)

```c
// DeepSeek Sparse Attention pattern for GQA layers at 256K
// Each query attends to: local window + global positions

typedef struct {
    int local_window;       // e.g., 4096 tokens
    int num_global;         // e.g., 128 global positions
    int global_stride;      // e.g., pick every 2048th token as global
} DSAParams;

float* dsa_attention(float* query, float* key_cache, float* value_cache,
                     int position, int seq_len, DSAParams p) {
    // Sparse set: current + local window + global positions
    // |<-local window->|<-skip->|<-global->|<-skip->|...|
    // O(L * (p.local_window + p.num_global)) instead of O(L^2)
    
    float* attention_scores = malloc((p.local_window + p.num_global) * sizeof(float));
    int idx = 0;
    
    // Local window: attend to last p.local_window positions
    for (int j = max(0, position - p.local_window); j < position; j++) {
        attention_scores[idx++] = dot_product(query, key_cache[j]);
    }
    
    // Global positions: attend to every p.global_stride-th position
    for (int j = 0; j < seq_len; j += p.global_stride) {
        if (j >= position - p.local_window) break;  // already in local window
        attention_scores[idx++] = dot_product(query, key_cache[j]);
    }
    
    // Softmax over sparse set
    softmax(attention_scores, idx);
    
    // Weighted sum of value vectors
    float* output = calloc(d_model, sizeof(float));
    idx = 0;
    for (int j = max(0, position - p.local_window); j < position; j++) {
        for (int d = 0; d < d_model; d++)
            output[d] += attention_scores[idx] * value_cache[j][d];
        idx++;
    }
    for (int j = 0; j < seq_len; j += p.global_stride) {
        if (j >= position - p.local_window) break;
        for (int d = 0; d < d_model; d++)
            output[d] += attention_scores[idx] * value_cache[j][d];
        idx++;
    }
    
    return output;
}
```

---

## PART 4: Unchanged Confirmed Design Decisions

The following WuBuText design choices are **validated and should NOT change**:

| Design Decision | Validation Source | Confidence |
|:---------------|:-----------------|:----------:|
| **256 experts, 8 active** | DeepSeek-V3 (identical config) | ★★★★★ |
| **30 SSM + 10 GQA layers (3:1 ratio)** | Gemma 3 (more local than global) | ★★★★★ |
| **SSM replaces local attention** | Gemma 3 (local attention for efficiency) | ★★★★★ |
| **Hybrid MoE architecture** | Qwen3 (dense + MoE variants) | ★★★★☆ |
| **C inference engine** | All papers (kernel-level optimization needed) | ★★★★★ |
| **Speculative decoding target** | DeepSeek-V3 MTP (self-speculative) | ★★★★☆ |

---

## PART 5: Risk & Trade-off Analysis

| Decision | Risk | Mitigation |
|:---------|:-----|:-----------|
| **SSM instead of local attention** | SSM quality compared to learned local attention | Monitor perplexity; fall back to sliding window attention if needed |
| **MTP (D>1) complexity** | Extra heads increase model size by ~D×vocab_dim | Keep D small (2-3); share trunk, only add output heads |
| **256 experts** | Memory overhead for expert weights | Use expert parallelism across GPUs |
| **Self-speculative decoding** | Draft quality from MTP heads may be lower than separate draft model | Compare against separate small draft model baseline |
| **RoPE 4x extrapolation** | Quality degradation at extreme positions | Validate with long-context benchmarks; consider progressive training as backup |

---

## PART 6: Quick Reference — Papers to Implementation Map

| WuBuText Component | Primary Source | Secondary Source | C File |
|:------------------|:---------------|:-----------------|:-------|
| MoE routing + gating | DeepSeekMoE (§3) | DeepSeek-V3 (§2) | `moe.c` |
| Load balancing | DeepSeek-V3 (§2.3) | — | `moe.c` |
| Shared experts | DeepSeekMoE (§2.2) | DeepSeek-V3 (§2.2) | `moe.c` |
| RoPE + extrapolation | Qwen2.5-1M (§3.1) | — | `attention.c` |
| Sparse attention (DSA) | DeepSeek-V3.2 (§2.1) | Qwen2.5-1M (§3.2) | `attention.c` |
| Chunked prefill | Qwen2.5-1M (§3.3) | — | `inference.c` |
| MTP + self-speculative | DeepSeek-V3 (§2.4) | Blog posts | `model.c`, `speculative.c` |
| Thinking/non-thinking mode | Qwen3 (§3) | — | `inference.c` |
| RL training | DeepSeek-R1 (§2) | — | Training pipeline |
| Progressive training | Qwen2.5-1M (§4) | — | Training pipeline |
| Local/global ratio validation | Gemma 3 (§2.1) | — | (design confirmed) |

---

*End of synthesis document. See individual vault paper files for full technical reports.*
