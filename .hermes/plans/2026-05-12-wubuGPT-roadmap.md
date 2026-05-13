# WuBuGPT 1000-Step Roadmap — WuBuNest + DeepSeek Fusion

> **End Goal:** Pure C end-to-end training + inference system that produces **WuBuText AI** (text) and **WuBuVision AI** (multi-modal). Uses wubu nested hyperbolic math (Lean-verified) to absorb existing LLM weights via Euclidean→Hyperbolic translation, then continues training with MLA+SparseMoE+WubuOptimizer. Karpathy-minGPT-style functional comparison.

---

## ═══════════ PART I: CURRENT STATE AUDIT ═══════════

### Assets Already Built

| Asset | Status | Lines | Notes |
|-------|--------|-------|-------|
| **C HashMind Training** | WORKING | ~6.4K | 36K param model, forward+backward+WubuOptimizer, saves .bin |
| **Lean Proofs (4 files)** | ✅ COMPILED | ~600 | PoincareBall, MobiusAdd, MLACompression, HyperbolicGyration |
| **NumPy WuBuNestGPT** | WORKING | ~2K | LatentAttention, MoE, hyperbolic gyration, trains on CORPUS.py |
| **JAX WuBuNestGPT v2** | WORKING | 454 lines | MLA+SparseMoE+gyration+quant, 63M params, KV cache |
| **llama.cpp models/** | INSTALLED | — | Qwen 3.5, DeepSeek V2/V3, DeepSeek R1 models available |
| **CORPUS.py** | EXISTS | 66K lines, 14.6MB | 10 protocol documents, training corpus |
| **OpenWebMath** | DOWNLOADED | 141MB | 20,000 docs for scaling pretrain |
| **WubuOptimizer** | ✅ IMPL | C code | Toroidal gradient mod + Adam bias correction |
| **C wubu_math.h** | ✅ | 134 lines | Poincaré exp/log, Möbius add, gyration, quaternions |
| **C nn_ops.h** | ✅ | 142 lines | GELU, softmax, cross-entropy, layer norm, matmul |
| **RTX 5050** | READY | — | 6.4GB VRAM, sm_89, WSL |

### Architecture Gaps vs. Full WuBuGPT

| Gap | Priority | Why Matters |
|-----|----------|-------------|
| No Euclidean→hyperbolic weight loader | P0 | Unlocks 99% skip of pretrain — load Qwen/DeepSeek, translate to ball |
| C hasn't integrated wubu_math.h into transformer | P0 | HashMind still uses standard attention, not MLA |
| No Sparse MoE in C | P1 | Without MoE, model can't scale past 36K params while fitting 6.4GB |
| No KV cache at all (C model is single-token) | P1 | Can't do efficient inference without it |
| No CUDA kernels in use | P1 | CPU-only training = 30 tok/s vs GPU = 2000+ tok/s |
| No post-training (RLHF/DPO/GRPO) | P2 | Needed for alignment |
| No vision encoder | P3 | For WuBuVision AI |
| No multi-modal fusion | P3 | Text+vision integration |

---

## ═══════════ PART II: 1000-STEP ROADMAP ═══════════

### P0 — CRITICAL PATH (Steps 001-150) — "Can Train a Real Model"

**Domain: Weight Translation + C Architecture Upgrade**

```
Goal: Load a real model, translate to hyperbolic space, train in C on GPU
Verification: Perplexity of 50 or lower on held-out CORPUS.py text
```

**Phase 0.1: Weight Loader — Euclidean → Hyperbolic Translation (Steps 001-040)**

001-010: Build GGUF reader in C (`src/gguf_reader.c` + `include/gguf_reader.h`)
  - Read GGUFv3 headers, metadata, tensor names
  - Support float32 and Q4_0/Q4_1/Q8_0 quantized tensors
  - Expose: `gguf_load(path)` → tensor dict with name→(data, shape, type)

011-020: Map DeepSeek V2/V3 tensor names to WuBu model layout
  - `wq.weight` → qkv_w (combined latent projection)
  - `wkv_a.weight` → latent KV compression (MLA key)
  - `wo.weight` → out_w
  - `ffn_gate/up/down.weight` → MoE expert weights
  - Write mapping table: `include/deepseek_tensor_map.h`

021-030: Implement Euclidean → Hyperbolic translation algorithm
  - For weight matrix W ∈ ℝ^{m×n}: take SVD, apply wubu_exp_map to singular vectors
  - Preserve attention structure: keep position embeddings linear (Euclidean)
  - Keep layer norms Euclidean (they work in both spaces)
  - C function: `wubu_translate_weights(W, W_out, c=1.0)`

031-040: Build `weight_loader.c` that:
  - Reads GGUF → tensor dict
  - Applies hyperbolic translation to all MLP weights
  - Copies attention weights as-is (or projects, configurable)
  - Writes HashMindModel .bin
  - Test: load Qwen 3.5 0.5B GGUF → translate → save → verify structure

**Phase 0.2: C Model Upgrade — MLA + MoE + KV Cache (Steps 041-100)**

041-050: Rebuild HashMindModel as scalable config (`include/wubu_model.h`)
  - `D_MODEL` 64→768, `N_LAYERS` 4→8, `N_HEADS` 4→12
  - Add MLA latent KV compression: `D_COMPRESSED = D_MODEL / 3` (DeepSeek ratio)
  - Add Sparse MoE: N_EXPERTS=8, TOP_K=2
  - Old model = HashMindMini compat layer

051-060: Implement MLA latent attention (`src/wubu_attention.c`)
  - `c_t^KV = W_DKV * h_t` (latent KV joint compression)
  - `q_t = W_Q * h_t` (full query, no compression)
  - KV cache stores c_t^KV only (d_c << n_h*d_h — DeepSeek trick)
  - On attention: up-project K,V from latent for dot product

061-070: Implement Sparse MoE (`src/wubu_moe.c`)
  - Top-2 expert routing with load balancing bias (DeepSeek V3 style)
  - Shared expert (always activated) + routed experts
  - Forward: for each token, compute gate logits → top-2 → weighted sum

071-080: Add KV cache to C model (`include/kv_cache.h`)
  - Ring buffer of latent KV vectors [CONTEXT_LEN, D_COMPRESSED]
  - Cache miss → recompute from scratch
  - Prefill support: process all tokens at once, build cache

081-090: Hook wubu_math.h into forward pass
  - Replace RoPE with Möbius gyration-based position encoding
  - After each attention head: apply gyration to V based on position delta
  - `gyration(v, pos_delta, c=1.0)` as position-dependent rotation

091-100: Test full forward/backward on RTX 5050
  - CUDA kernels for: matmul (use llama.cpp's ggml-cuda dispatch)
  - Or: implement naive CUDA kernels for attention + MoE
  - Verify: forward pass produces stable logits on CORPUS.py sample

**Phase 0.3: GPU Training — Full Pipeline (Steps 101-150)**

101-110: Port WubuOptimizer to CUDA
  - Toroidal gradient mod kernel (wrap [-π,π])
  - Adam momentum + variance on GPU
  - Weight decay kernel

111-120: Build training pipeline on RTX 5050
  - Batch size: 4-8 sequences (fit 6.4GB VRAM)
  - Mixed precision: float32 master weights, float16 for matmul
  - Gradient accumulation: fake large batch via 4 micro-batches

121-130: Train on CORPUS.py (first real run)
  - 10 epochs, track loss curve
  - Compare perplexity against NumPy reference
  - Should match within 0.5% if math is correct

131-140: Initial weight translation test
  - Load Qwen 3.5 0.5B → translate → continue training on CORPUS.py
  - Measure: perplexity at step 0 (should be <100 if translation works)
  - Training loss curve should drop from translated baseline

141-150: Optimization & debugging
  - Profile: where is time spent? (matmul? attention? MoE?)
  - Fix any gradient issues (NaN detection, gradient clipping)
  - Verify backward pass with finite differences
  - Save first checkpoint: `wubu_gpt_corpus_run1.bin`

---

### P1 — CORE FEATURES (Steps 151-400) — "LLM That Actually Works"

**Domain: Scaling + Data + Inference**

**Phase 1.1: Scale to DeepSeek-sized Model (Steps 151-200)**

151-160: Memory optimization for RTX 5050 (6.4GB)
  - Gradient checkpointing: recompute activations during backward
  - Mixed precision: fp16 storage, fp32 for loss-critical ops
  - Activation offloading: spill to system RAM during long sequences

161-170: Increase to real-scale model
  - D_MODEL=1024, N_HEADS=16, D_COMPRESSED=512
  - N_LAYERS=12 (can fit in ~3.5GB with fp16)
  - Sparse MoE: 16 experts, top-4, shared=2

171-180: Parallel data loading
  - Background thread loads CORPUS.py + OpenWebMath
  - Tokenize on CPU, prefetch to GPU
  - Streaming: don't hold entire dataset in memory

181-190: Distributed training (optional — single GPU for now)
  - If second GPU available: data parallelism
  - Otherwise: single GPU, focus on throughput

191-200: Training stability
  - Learning rate warmup (first 1000 steps)
  - Cosine schedule to 0 over total steps
  - Gradient clipping: max norm = 1.0
  - Weight decay: 0.1 for all non-bias/norm params

**Phase 1.2: Full Pretrain Run (Steps 201-280)**

201-220: Prepare real dataset
  - Convert CORPUS.py → flat training text (strip Python, keep narratives)
  - Pre-tokenize OpenWebMath with WuBu tokenizer (rolling hash + ASCII)
  - Merge into training binary format (fast random access)

221-240: Run pretraining
  - 10x the steps of NumPy run (100K+ tokens)
  - Log every 1000 steps: loss, perplexity, samples
  - Save checkpoint every epoch (every ~20K tokens)

241-260: Mid-training evaluation
  - Held-out validation loss
  - Compare against: same train setup with standard GPT (no hyperbolic)
  - Metric: does hyperbolic model learn faster (lower loss at same step count)?
  - Generate text samples: are they coherent?

261-280: Full training completion
  - Train to convergence (loss flattening)
  - Save final model: `wubu_gpt_v1.bin`
  - Generate test samples across prompts

**Phase 1.3: Inference System (Steps 281-330)**

281-290: Build inference-only executable
  - `wubu_infer --model wubu_gpt_v1.bin --prompt "..." --tokens 256`
  - Temperature, top-k, top-p sampling
  - Streaming output (print token by token)

291-300: KV cache optimization
  - Quantize cached KV latents to fp8 (DeepSeek V3 turbo style)
  - Per-block scaling factors for fp8
  - Benchmark: fp16 vs fp8 cache (quality vs memory)

301-310: Batch inference server
  - Accept multiple prompts
  - Process batched KV cache
  - HTTP server for API access

311-320: Performance benchmarks
  - Tokens/sec: compare C model vs llama.cpp with same size
  - GPU utilization: is matmul bound? memory bound?
  - Optimize bottlenecks

321-330: Model comparison
  - Run minGPT + same data → compare loss curves
  - Run NumPy WuBuNestGPT → compare exactly
  - Publish results in benchmark table

**Phase 1.4: Fine-tuning Pipeline (Steps 331-400)**

331-340: Fine-tuning infrastructure
  - Load pretrained model
  - Load fine-tuning dataset (instruction pairs)
  - Train with lower LR (1e-5) on instruct data

341-360: Instruction tuning
  - Format: `[INST] {prompt} [/INST] {response}`
  - Train on Alpaca-like instruction data
  - Evaluate: does model follow instructions?

361-380: Reinforcement learning from human feedback
  - Build reward model (small transformer trained on human preferences)
  - PPO implementation in C (can piggyback on existing forward/backward)
  - Run: 1 epoch of PPO after instruct tuning

381-400: GRPO (Group Relative Policy Optimization) — DeepSeek-R1 style
  - No critic network needed
  - Sample N responses per prompt, compute rewards, normalize within group
  - Train policy with KL penalty
  - Implementation: pure C, no Python

---

### P2 — ENHANCEMENT (Steps 401-600) — "Production Quality"

**Phase 2.1: CUDA Kernel Ecosystem (Steps 401-450)**

401-420: Flash Attention-style kernel for hyperbolic attention
  - Tiled matmul with online softmax
  - Handle Möbius gyration position encoding inside kernel
  - Benchmark vs naive implementation

421-440: Sparse MoE kernel
  - Expert parallelism: each expert assigned to different SM
  - Dynamic dispatch: tokens route to correct expert
  - Avoid warp divergence with balanced routing

441-450: Quantization kernels
  - FP8 block quantization (Turbo style)
  - INT4 weight-only quantization (for inference)
  - Calibration: find optimal scaling factors

**Phase 2.2: Data Pipeline + Multi-Epoch Training (Steps 451-520)**

451-470: Large-scale data preparation
  - Download and preprocess Pile / C4 / FineWeb
  - Deduplicate with rolling hash
  - Tokenize and shard into .bin files

471-500: Multi-epoch training
  - Train on 10B+ tokens (may take days on RTX 5050)
  - Progressive learning rate schedule
  - Curriculum: short sequences first, then long

501-520: Early stopping & model selection
  - Track validation loss
  - Save best model (lowest validation loss)
  - Generate comprehensive test samples

**Phase 2.3: Advanced Architectures (Steps 521-580)**

521-540: Add Mamba-style state space layers
  - Mix SSM + attention (hybrid — Qwen 3.5 style)
  - SSM for efficiency on long sequences
  - Attention for recall on short sequences

541-560: Hyperbolic native attention (no Euclidean fallback)
  - Full attention in Poincaré ball
  - Möbius attention score computation
  - Gyration-based position encoding

561-580: Multi-query / grouped-query attention variants
  - GQA: 2 KV heads per 8 query heads
  - Benchmark: quality vs speed tradeoff

**Phase 2.4: Optimization & Hardening (Steps 581-600)**

581-590: Numerical stability
  - All operations tested on 1M+ step training
  - No NaN, no gradient explosion
  - Deterministic: same seed → same result

591-600: C code refactor
  - Modular build system
  - Unit tests for each module
  - Documentation for API

---

### P3 — ECOSYSTEM + MULTI-MODAL (Steps 601-1000) — "WuBuVision AI"

**Phase 3.1: WuBuVision AI — Vision Encoder (Steps 601-700)**

601-620: Design vision architecture
  - Vision transformer (ViT) with hyperbolic patch embedding
  - Image → patches → Poincaré ball embedding → transformer
  - Position encoding via Möbius gyration (as in text)

621-650: Build vision encoder in C
  - Patch extraction + linear projection
  - Hyperbolic vision transformer blocks
  - [CLS] token for classification

651-670: Train vision encoder
  - ImageNet-1k classification (or smaller — CIFAR-100 first)
  - Transfer learning: load pretrained weights → hyperbolic translate → fine-tune

671-700: Vision feature extractor
  - Given image → output feature vectors in Poincaré ball
  - Compatible with text embeddings for multi-modal

**Phase 3.2: Multi-Modal Fusion (Steps 701-800)**

701-730: Fusion architecture
  - Text tower + vision tower → shared hyperbolic space
  - Cross-attention between modalities
  - Router: decide which modality to attend to

731-760: Train multi-modal model
  - Image captioning: image → text
  - Visual Q&A: image + question → answer
  - Contrastive: image-text matching

761-800: Multi-modal inference
  - Accept image + text prompt
  - Generate text response with visual understanding
  - Streaming output

**Phase 3.3: Post-training + Alignment (Steps 801-880)**

801-830: RLHF at scale
  - Reward model trained on human preferences
  - PPO training on instruction data
  - Evaluation: helpfulness + harmlessness

831-860: DeepSeek-R1 style reasoning
  - Chain-of-thought training
  - Process reward model (step-by-step verification)
  - Self-consistency decoding

861-880: Safety alignment
  - Red-teaming: find failure modes
  - Safety filter: detect harmful prompts before generation
  - Constitutional AI: self-critique + revision

**Phase 3.4: Deployment + Ecosystem (Steps 881-950)**

881-900: HTTP/API server
  - REST API for text generation
  - WebSocket for streaming
  - OpenAI-compatible API

901-920: Model distribution
  - Quantized models for CPU inference
  - Model zoo: download pre-trained WuBu models
  - Versioned releases

921-940: Developer tools
  - Python bindings (via CFFI or pybind11)
  - Jupyter notebook examples
  - Documentation website

941-950: Community
  - GitHub repo with clear contribution guide
  - Discord for discussion
  - Benchmark leaderboard

**Phase 3.5: Research + Publishing (Steps 951-1000)**

951-970: Paper writing
  - "WuBuGPT: Training LLMs in the Hyperbolic Ball"
  - Include: Lean proofs, architecture, benchmark results, comparisons
  - Submit to NeurIPS / ICML

971-990: Ablation studies
  - Hyperbolic vs Euclidean: same model, same data
  - Möbius gyration vs RoPE vs no position encoding
  - MoE vs dense: parameter-matched comparison
  - Lean-proven math as competitive advantage

991-1000: Release v1.0
  - Tag: v1.0.0
  - Release notes with benchmark results
  - Pre-trained model download links
  - Congratulations — WuBuGPT is real

---

## ═══════════ PART III: DEVIL'S ADVOCATE ═══════════

### 🛑 Stop. Read this before spending one cycle on the roadmap above.

I am going to **attack every assumption** in the roadmap and the WuBuGPT vision. This isn't negativity — it's preventing you from wasting months.

---

### 🔴 Critical: "Euclidean → Hyperbolic Weight Translation" is UNPROVEN

**Claim:** We can load Qwen/DeepSeek weights, apply a hyperbolic translation, and continue training without catastrophic loss of information.

**Reality check:** This is NOT a known operation. Nobody has done this. The properties of pre-trained weights in Euclidean space do not transfer to hyperbolic space:

1. **SVD doesn't preserve geometry.** The approximation `W ≈ U·Σ·Vᵀ` → apply exp map to U → the resulting matrix is no longer an isometry. The output distribution changes unpredictably.

2. **Layer norms break.** A layer norm computed in Euclidean space assumes Gaussian-ish distributions. After hyperbolic projection, the distribution is constrained to the ball surface — layer norm statistics are completely wrong.

3. **Logits will be garbage.** Even if per-layer representations survive, the final LM head projects to a Euclidean vocabulary. A hyperbolic hidden state projected through a Euclidean vocabulary matrix produces meaningless logits.

4. **The P0 claim "perplexity < 100" is wishful thinking.** Best case: perplexity ~500 (random guessing). Realistic case: model generates NaN or constant outputs.

**Devil's counter-proposal:** Drop the "load pretrained → translate" idea for now. Train WuBuGPT from scratch in hyperbolic space. If you need pretrained weights, **keep 99% of computation in Euclidean space** and only apply hyperbolic gyration for position encoding (which is what the JAX code does and what works).

---

### 🔴 Critical: "Pure C training" is a massive time sink with no benefit

**Claim:** The end-to-end system runs in pure C for "efficiency".

**Reality:** You already have a working JAX implementation (`wubu_nest_gpt_v2.py`) with 63M params, MLA, MoE, and hyperbolic gyration. The C code has a **36K param model** that barely fits in L1 cache.

| Aspect | JAX | C (current) |
|--------|-----|-------------|
| Params | 63M | 36K |
| Training speed | 2000+ tok/s (GPU) | 30 tok/s (CPU) |
| Forward/backward | XLA-optimized, auto-parallel | Manual loops, single-threaded |
| Autograd | Automatic | Manual (bug-prone) |
| CUDA support | Native | Via ggml-cuda (unused) |
| Lines of code | ~1K | ~14K (and growing) |

**The C code is NOT competitive** for training. Even with CUDA kernels, you're rewriting everything XLA does for free.

**If you want C:** Use llama.cpp's ggml as a backend. Write the model architecture as a ggml graph (like the DeepSeek V2 implementation in `llama.cpp/src/models/deepseek2.cpp`). Then you get:
- Free CUDA acceleration
- Free CPU fallback
- Free quantization (Q4_0, Q8_0, FP16)
- Free KV cache management
- 262 lines vs 488 lines for attention alone

**Devil's counter-proposal:** Kill standalone C training. Build WuBuGPT as a **ggml graph builder** following the DeepSeek V2 pattern in `llama.cpp/src/models/deepseek2.cpp`:
- 262 lines for MLA attention vs 488 lines in C
- 386 lines for full model vs ~600+ lines in standalone C
- Free: GPU, quantization, KV cache, batch inference

---

### 🟡 High Risk: "Hyperbolic is Better" — Where is the evidence?

**Claim:** Hyperbolic geometry improves LLM training (faster convergence, better representations, etc.)

**Evidence base:**
- ✅ Lean proofs: Möbius addition preserves the ball (verified)
- ✅ JAX code: forward pass works, outputs finite numbers
- ⬜ Training curves: NO comparison data exists. Is the hyperbolic model learning FASTER than Euclidean?

**What's missing:**
- Ablation: same architecture, same training, different position encoding (RoPE vs gyration vs none)
- Ablation: same architecture, same training, Euclidean attention vs hyperbolic attention
- Any evidence that hyperbolic spaces capture hierarchical structure better than a deep transformer already does

**Hypothesis that needs testing:** A 6-layer hyperbolic transformer learns at the SAME RATE as a 6-layer Euclidean transformer. Möbius gyration is just fancier RoPE.

**Devil's counter-proposal:** Before step 100 of the roadmap, run the ablation:
1. Train standard GPT (NumPy)
2. Train WuBuNestGPT (NumPy) with identical params
3. Compare loss curves at steps 100, 500, 1000, 5000
4. If hyperbolic is WORSE than Euclidean, drop the whole approach
5. If hyperbolic is EQUAL, the complexity has no benefit
6. Only proceed if hyperbolic is CLEARLY BETTER (p < 0.01 on t-test)

---

### 🟡 High Risk: "Lean proofs → Better AI" is a non-sequitur

**Claim:** Formal proofs of hyperbolic geometry properties make WuBuGPT more reliable.

**Reality:** The Lean proofs verify that:
1. `exp_0^c(log_0^c(y)) = y` (so the math is self-consistent)
2. Möbius addition preserves the ball (outputs stay valid)
3. `H*(W_K - W_DKV*U_K) = H*W_K - H*W_DKV*U_K` (matrix factorization is correct)
4. In 1D, gyration is identity

**These are the EASIEST properties to verify.** The hard properties — convergence guarantees, generalization bounds, approximation capacity — are NOT proven and may not even be true.

The gap between "the math is sound" and "this trains a better LLM" is the entire field of deep learning.

---

### 🟡 Medium Risk: RTX 5050 6.4GB is SEVERELY limiting

| Model Size | Memory (fp32) | Memory (fp16) | Fits 6.4GB? |
|------------|---------------|---------------|--------------|
| 36K params (current) | 0.14 MB | 0.07 MB | ✅ Easily |
| 63M params (JAX v2) | 252 MB | 126 MB | ✅ |
| 350M params (Qwen 0.5B) | 1.4 GB | 700 MB | ✅ (tight with activations) |
| 1.5B params | 6 GB | 3 GB | ❌ (no room for activations) |
| 7B params | 28 GB | 14 GB | ❌ |

Maximum realizable model on RTX 5050: **~300M params** with fp16 + gradient checkpointing.

That's ~50x smaller than Qwen 3.5 0.5B, which needs ~1GB for weights alone.

**For comparison:** GPT-2 Small = 124M params. So you can train a GPT-2 class model. Not bad, but far from DeepSeek V3 scale.

---

### 🟢 Low Risk: "WubuOptimizer with toroidal gradient" — actually interesting

The toroidal gradient mod (wrap gradient into [-π,π] before momentum) is **novel and likely beneficial**. The intuition:
- Gradients in hyperbolic space have bounded norm (the ball is compact)
- Toroidal wrap prevents momentum from accumulating in one direction indefinitely
- This could stabilize training for deep hyperbolic networks

**This is the most defensible claim in the whole project.** Keep this, prioritize proving it works.

---

### 🟢 Low Risk: "C inference is good" — yes, for deployment

Once a model is trained, C inference on CPU is valuable:
- Edge deployment
- Privacy-preserving local inference
- No Python dependency

But this is a **P3 concern** — you need a trained model first. Do NOT invest in C inference until the model works.

---

## ═══════════ PART IV: REVISED MINIMAL PLAN ═══════════

### What to ACTUALLY do, given the devils' advocacy

**Step 1 (1 week): Run the ablation**
- Take `wubu_nest_gpt_numpy.py` — it already trains!
- Add Euclidean-only version (same architecture, replace Poincaré with standard attention)
- Run both on CORPUS.py for 5000 steps
- If hyperbolic loses, drop hyperbolic attention (keep only gyration for position encoding)
- IF AND ONLY IF hyperbolic wins: proceed

**Step 2 (1-2 weeks): Port to llama.cpp ggml**
- Follow `deepseek2.cpp` pattern (MLA attention)
- Replace RoPE with Möbius gyration (this is the win)
- Use ggml-cuda for free GPU acceleration
- Reuse existing gguf weight loading
- Result: WuBuGPT as a ggml model, training on GPU

**Step 3 (1 week): Train + Compare**
- Train WuBuGPT (ggml) on CORPUS.py
- Train equivalent Euclidean model (also ggml)
- Compare loss curves, generation quality, training speed
- Publish: one chart showing hyperbolic beats Euclidean or not

**Step 4 (2 weeks): Scale + Fine-tune**
- If hyperbolic wins: scale to 300M params, train on OpenWebMath
- Fine-tune on instruction data
- Compare: benchmark results with open-source models of same size

**Step 5 (long-term): WuBuVision AI**
- Only after text model is validated
- Vision encoder as separate ggml model
- Cross-attention for multi-modal

---

### The ONE metric that matters

Not code written, not proofs compiled, not lines of C code.

**The metric: Is the hyperbolic model BETTER than Euclidean, measured by:**
1. Lower validation perplexity at same step count
2. Better sample quality (human eval)
3. Better parameter efficiency (more information per param)

If you can't answer "yes" to #1 within 2 weeks, the roadmap is wrong.

---

> **Last updated:** 2026-05-12
> **Scope:** WuBuGPT Text (P0-P2) + WuBuVision AI (P3)
> **Hardware:** RTX 5050 6.4GB, WSL, CUDA 12.x
> **Active skills:** writing-plans, subagent-driven-development, test-driven-development
