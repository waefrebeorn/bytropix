# Plan — Phase 28j+: Extended Roadmap with Vault Cross-Reference (DA v12+vault)

## 🔴 P0: Hidden State Divergence (DA C6 — VERIFIED)
GPU hidden cos-sim -0.0036 vs CPU with FORCE_CPU_SSM.

**Step 1:** Isolate MoE vs GQA
- `src/wubu_model.c:636-639` — remove `layer->moe.gpu_ctx = (void*)model;`
- Rebuild gen_text_gpu, run with FORCE_CPU_SSM + DUMP_HIDDEN
- If hidden fixes → MoE kernel is buggy → debug `gpu_moe_kernel.cu`
- If still broken → GQA is also buggy → debug `wubu_model_gpu.cu` GQA section

**Step 2:** If MoE — insert expert-level dequant dump, compare gate/up/down outputs
**Step 3:** If GQA — compare GPU GQA output vs CPU GQA for same input

## 🟡 P1: GPU SSM + MTP + Hedged Spec Decode
1. Fix forward_full GPU SSM divergence (gpu_ssm_recurrence.cu)
2. Fix GPU SSM C>1 prefill (cuBLAS error 13)
3. Build + test gen_text_mtp with MTP model (verify 83% acceptance at 2 drafts)
4. **N-way hedged speculative decode** (vault/tailslayer/) — parallel draft verification across GPU SMs using tailslayer DRAM-channel hedged-read pattern

## 🟡 P2: Vision Integration + Encoders
1. Build test_vision_real, verify 3D ViT encoder output
2. Wire full vision→text multi-modal pipeline
3. **Hamiltonian KV cache compression** (vault/hamilton/) — 10× compression, 62% compression ratio, ~3% overhead. Already has CPU prototype, needs CUDA port.
4. **Quaternion/Geodesic attention** (vault/hamilton/) — alternative attention mechanism, partial CUDA port exists

## 🟢 P3: Feature Cream — Vault-Sourced
| Feature | Vault Source | Priority |
|---------|-------------|----------|
| Sigmoid gating + load balancing | deepseek-papers/ (DeepSeekMoE) | High |
| Chunked prefill (3-7x speedup) | qwen-papers/ (Qwen2.5-1M) | High |
| RoPE extrapolation 4x | qwen-papers/ | High |
| Sparse attention (NSA) | deepseek-papers/ (DeepSeek V3.2) | High |
| GPU RMSNorm + SiLU + gated norm kernels | Engineering (own) | High |
| **WuBuSparseAttention** (dual-memory RAS, O(n·k)) | vault/attention/ | Mid |
| **Topological Sequence Model** (Conv1D+Poincaré, O(n)) | vault/attention/ | Mid |
| **Entropix Sampler** (Dirichlet dynamic sampling) | vault/attention/ | Mid |
| **Rolling Hash Attention** (SimpleHash C port) | vault/hash-mind/ | Mid |
| **Tri-Cameral Hyperbolic Attention** | vault/attention/ | Low |

## 🟢 P4: Training Pipeline — Vault-Sourced
| Feature | Vault Source | Priority |
|---------|-------------|----------|
| CUDA training kernels + FSDP + GRPO RL | plan.md (Phase 32 original) | High |
| **Q-Controller** (Q-learning LR scheduler) | vault/optimizers/ | Mid — directly helps training stability |
| **PID Lambda Controller** (loss-weight balancing) | vault/optimizers/ | Mid — adaptive loss control |
| **Pure C training pipeline** reference | vault/c-training/ | Low — demonstrates feasibility |
| **Dual-Agent Q-Learning Training** | vault/hash-mind/ | Low |
| **HAKMEM Optimizer** | vault/draftPY/ | Low |

## 🔵 P5: Multi-Modal Expansion — Vault-Sourced
| Feature | Vault Source | Priority |
|---------|-------------|----------|
| **Symmetric Geometric Autoencoder** (image latent) | vault/encoders/ | Mid — complements P2 vision |
| **Topological Quantum Autoencoder** (3-float compression) | vault/encoders/ | Low |
| **Text-to-Image Generation** (VQ-VAE + Conductor Transformer) | vault/phase3/, draftPY/, diffusion/ | Future |
| **Diffusion Models** (HGA-UNet, Funnel Diffusion) | vault/diffusion/ | Future |
| **Audio Synthesis** (WubuSynth + EnCodec) | vault/audio/ | Future |
| **GAAD (Golden Aspect Adaptive Decomposition)** | vault/draftPY/ | Future |
| **Spectral Transformer (SpecTrans)** | vault/draftPY/ | Future |
| **HypCD/BSFIN Hyperbolic Compression** | vault/draftPY/ | Future |

## 🔵 P6: Infrastructure — Vault-Sourced
| Feature | Vault Source | Priority |
|---------|-------------|----------|
| **Lean 4 Formal Verification** of hyperbolic math | vault/lean-proofs/ | Low — assurance, not product |
| **ETP (Embedding-to-Physics) Training** | vault/draftPY/ | Future |

## Vault Cross-Reference Summary
- **18 vault READMEs examined** via subagent
- **13 vault areas contain capabilities NOT on roadmap**
- **23 missing items identified** (M1-M23), categorized into P3-P6 above
- **4 vault areas fully mapped**: deepseek-papers, qwen-papers, synthesis.md, deepmind-2026
- **1 vault archival**: bins/
- **1 already implemented**: theory/ (hyperbolic math in C/CUDA)

## Key Vault Directories (for context)
| Vault | Content | Roadmap Coverage |
|-------|---------|-----------------|
| `vault/attention/` | 4 attention variants + Entropix sampler | P3 (partial: sparse attn + hyperbolic) |
| `vault/hamilton/` | Quaternion attention, KV cache compression | P2 (KV cache comp) + P3 (quaternion attn) |
| `vault/hash-mind/` | WuBuMind JAX, rolling hash attention | P3 (rolling hash) + P4 (Q-learning) |
| `vault/optimizers/` | Q-Controller, PID Lambda | P4 (training) |
| `vault/tailslayer/` | Hedged speculative decode | P1 (MTP complement) |
| `vault/encoders/` | Geometric + quantum autoencoders | P5 (multi-modal) |
| `vault/phase3/`, `diffusion/`, `audio/` | Text-to-image, diffusion, audio synth | P5 (future) |
| `vault/lean-proofs/` | Formal verification | P6 (infrastructure) |
| `vault/c-training/` | Pure C training reference | P4 (reference) |
| `vault/draftPY/` | 40+ experimental scripts | P5/P6 (scattered) |
