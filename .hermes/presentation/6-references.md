# 6. References

**Purpose:** Annotated bibliography of the primary sources informing WuBuText AI. All claims about methodology, architecture, and performance in this presentation trace back to these papers. Language is intentionally conservative — this is a research prototype, not a production system.

---

## 1. Token-Superposition Training (TST)

**Bowen Peng, Théo Gigant, Jeffrey Quesnelle (Nous Research)**
[*Efficient Pre-Training with Token Superposition*](https://arxiv.org/abs/2605.06546)
arXiv:2605.06546, May 2026

- **Core contribution:** Drop-in pre-training acceleration that bags `s` contiguous tokens into one average embedding, runs the forward pass on `1/s` the sequence length, and uses Multi-Hot Cross-Entropy (MCE) loss over targets in the bag. A recovery phase (~75% of steps) returns to standard next-token CE loss, carrying over weights from superposition.
- **Results reported:** Up to 2.5× speedup at equal loss on a 10B A1B MoE model (4,768 B200-hours vs. 12,311 baseline). Validated at 270M–10B scales.
- **Application to this project:** Selected as the Phase 3 training methodology. Our model (Qwen3.6-35B-A3B, 3B active parameters) is similar in scale to the 10B A1B test case. TST requires no architecture changes — only modified embedding lookup and loss computation. Target: 2×+ speedup on consumer GPU.
- **Known limits:** Results are from a single lab; independent reproduction is pending. Bag size sensitivity at s > 8 needs local validation.

---

## 2. DeepSeek-V2 — Multi-Head Latent Attention (MLA)

**DeepSeek-AI**
[*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*](https://arxiv.org/abs/2405.04434)
arXiv:2405.04434, May 2024

- **Core contribution:** MLA compresses the KV cache into a low-dimensional latent vector `c_t = W_DKV @ [k_t; v_t]`, then decompresses on read. Decoupled RoPE — a separate small rotary dimension — allows positional encoding without breaking the latent compression.
- **Relevance to WuBu:** MLA's compression scheme is a natural fit for hyperbolic context. The latent vector `c_t` lives in a low-dimensional space that could be mapped directly into the Poincaré ball. Not currently implemented; noted as a candidate for Phase 4+ efficiency improvements.

---

## 3. DeepSeek-V3 — MLA + MoE at Scale

**DeepSeek-AI**
[*DeepSeek-V3: A Groundbreaking Large Language Model*](https://arxiv.org/abs/2412.19437)
arXiv:2412.19437, December 2024

- **Core contribution:** Scales MLA to 671B total parameters (37B active) with aux-loss-free load balancing and multi-token prediction (MTP). Demonstrated that latent compressed attention + fine-grained MoE can match or exceed dense transformer quality at lower inference cost.
- **Relevance to WuBu:** Our target architecture (Qwen3.6-35B-A3B) uses a similar paradigm — hybrid SSM/GQA attention with 256 experts, 8 active per token. DeepSeek-V3's load-balancing and MTP strategies are reference points for Phase 4 MoE implementation.

---

## 4. Qwen3.6-35B-A3B — Architecture Reference

**Qwen Team (Alibaba Cloud)**
[*Qwen3.6-35B-A3B model card*](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
Hugging Face, May 2026

- **Architecture:** 40-layer hybrid: 30× Gated DeltaNet (SSM) + 10× GQA full attention, 3:1 repeating pattern. Hidden dimension 2048, 256 experts (8 routed + 1 shared), 248K vocabulary, 262K native context.
- **SSM layer:** Gated Delta Net with Conv1d (kernel=4), 16 query heads (d_state=128), 32 value heads (d_state=128), dt_rank=32, inner_dim=4096.
- **GQA layer:** 16 query heads, 2 KV heads (8:1 ratio), head_dim=256, MRoPE with sections [11, 11, 10], partial_rotary_factor=0.25.
- **Relevance to WuBu:** This is the weight source. All 733 GGUF tensors are extracted, mapped into the Poincaré ball via exponential map, and executed in pure C + CUDA. The 2048 hidden dimension and 3B active parameters fit within 8 GB VRAM with quantization.

---

## 5. Embedding Grafting — Euclidean → Poincaré Mapping

**Internal project work (May 2026)**

- **What it does:** Extracts `token_embd.weight` (248,320 × 2048, Q5_K) and `output.weight` from the Qwen3.6 GGUF file. Dequantizes to f32, then maps each Euclidean embedding vector `v` into the Poincaré ball of radius R = 0.956 via the exponential map: `exp_map(v) = tanh(||v||_2 / R) · v / ||v||_2`.
- **Validation:** 95% nearest-neighbor preservation after mapping (preliminary, on a sample of 10K tokens). 73 zero-norm special tokens correctly positioned at the origin.
- **Applicable reference:** The exponential/log map formulas follow the standard treatment in Ganea et al., *Hyperbolic Neural Networks* (NeurIPS 2018, arXiv:1805.09112). Poincaré ball radius R was chosen empirically such that 99.9% of mapped embedding norms fall below 0.99 (avoiding boundary instability).

---

## 6. SSM Recurrence — Gated Delta Net (from qwen3next.cpp)

**Source analysis of llama.cpp `qwen3next.cpp` (internal, May 2026)**

- The SSM recurrence in Qwen3.6 is a **Gated Delta Network** (not Mamba2 scan). The per-head recurrent update is:
  ```
  h[t] = h[t-1] · exp(gate[t]) + K[t] · (V[t] − h[t-1] @ K[t]) · beta[t]
  output[t] = h[t] @ Q[t]
  ```
- Exponential decay via `exp(−exp(ssm_a) · softplus(dt_alpha + dt_bias))`, with learnable `ssm_a[32]` and `dt_bias[32]`.
- Parallel chunked implementation uses triangular matrix operations. Our C/CUDA port uses the sequential autoregressive form (n_tokens == 1), which is the correct path for inference and training.
- **Limitation:** The chunked parallel form could offer training speedups; not yet implemented in our port.

---

**Note:** This is a working bibliography. As the project progresses through Phases 4–6, additional references (DeepSeek-V3.2 DSA, Qwen3.6 vision encoder papers, and formal hyperbolic geometry sources) will be added. All external claims about model performance, speedup factors, and loss numbers are as reported by the original authors — they have not been independently reproduced in this project.
