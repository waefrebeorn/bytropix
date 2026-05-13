# Phase 3: Training Loop — with Token-Superposition Training (TST)

**Goal:** Full training loop for WuBuText AI using TST for up to 2.5x speedup on pre-training.

**Paper:** [Efficient Pre-Training with Token Superposition](https://arxiv.org/abs/2605.06546) — `.hermes/references/TST_TOKEN_SUPERPOSITION.md`
**Authors:** Bowen Peng, Théo Gigant, Jeffrey Quesnelle (Nous Research)
**Validated on:** 270M, 600M, 3B dense + 10B A1B MoE (same architecture class as our model)

**Depends on:** Phase 1 ✅, Phase 2 ✅ (all 40 layers in C + CUDA, GPU verified 9.53 tok/s)

## Architecture Overview

```
Input IDs [B, T]
    ↓
Token Embedding [B, T, 2048]  ← Qwen3.6 extracted (frozen or trainable)
    ↓
RMSNorm (pre-embedding norm, blk.N.attn_norm)
    ↓
exp_map → Poincaré Ball (R=0.956)
    ↓
40× Blocks:
  Layer type determined by layer_types[] in config.json
  Type A (linear_attention, 30/40):  [Gated DeltaNet(SSM)] → [MoE]
  Type B (full_attention, 10/40):    [GQA] → [MoE]
    ↓
RMSNorm (final)
    ↓
log_map → Euclidean
    ↓
LM Head [2048, 248320]  ← Qwen3.6 extracted (shared weight possible? config says tie_embeddings=false)
    ↓
Softmax → Cross-Entropy Loss (+ optional MTP loss)
```

**Key correction from Qwen3.5 analysis:** The attention is NOT "Gated DeltaNet" as described
in any paper — it's a **structured SSM (Mamba2-style)** with `ssm_a`, `ssm_dt`, `ssm_conv1d`,
`ssm_alpha`, `ssm_beta` tensors per block. The "Gated DeltaNet" name in the model card is
descriptive, not a reference to the academic DeltaNet paper.

## Step 3.1: Data Pipeline

**Files:** `src/data_pipeline.c`, `include/data_pipeline.h`

**Tokenizer STATUS: ✅ DONE (May 13, 2026)**
The C BBPE tokenizer (`src/wubu_tokenizer.c`, 931 lines) matches HF Qwen3.6 EXACTLY for all text types:
- English ASCII (letters, numbers, contractions) ✅
- CJK characters (Chinese, Japanese) ✅
- BPE merge algorithm fixed (gap-safe indexing) ✅
- Byte encoder sourced from HF `tokenizer.json` (not hardcoded GPT-2) ✅
- 7 test cases verified against Python reference ✅

**Data pipeline still TODO:** The CORPUS.py data (66K lines) needs to be tokenized to binary format for C consumption. Options:
1. **Pre-tokenize to binary** via a one-time Python run → `.bin` file of token IDs
2. **Subprocess bridge** — C calls Python tokenizer via pipe (slow, interim)
3. **In-memory generation** — Synthetic training data (current train_stub.c approach)

### Tokenizer Details from GGUF
```
tokenizer.ggml.model: "gpt2"
tokenizer.ggml.pre: "qwen35"          ← Qwen-specific preprocessing
tokenizer.ggml.eos_token_id: 248046
tokenizer.ggml.bos_token_id: 248044
tokenizer.ggml.padding_token_id: 248055
tokenizer.ggml.add_bos_token: false
```

**Note:** `tokenizer.ggml.pre = "qwen35"` means there's Qwen-specific preprocessing.
Need to check what `qwen35` pre-tokenizer does in llama.cpp.

## Step 3.2: Loss Function — TST Multi-Hot CE (superposition) + Standard CE (recovery)

TST uses two-phase loss:

### Phase A: Superposition (ratio `r` of steps)
- Bag `s` contiguous tokens → average their embeddings → forward pass on `L/s` tokens
- Labels: shift left by `s-1`, split into non-overlapping bags of size `s`
- MCE loss = `(1/s) × Σ CE(pred, target[i])` for i in [0..s)
- Each TST step processes `s×` more tokens at same FLOPs (increase seq_len by `s`)
- No architecture changes needed

### Phase B: Recovery (ratio `1-r` of steps)
- Remove TST code entirely, standard CE next-token prediction
- Weights carry over from superposition phase checkpoint

### TST Params
| Param | Dense (270M-3B) | MoE (10B A1B) | Our Model |
|-------|-----------------|---------------|-----------|
| Bag size `s` | 6-8 | 16 | **TBD (start 8)** |
| Step ratio `r` | 0.3 | 0.25 | **0.25** |
| Speedup (equal-loss) | ~1.5x | **2.58x** | Target: **2x+** |

### Optional: MTP after recovery phase
1 additional head (shares embeddings), loss = CE + 0.3×CE_MTP. Only used during recovery phase after TST is removed.

## Step 3.3: WuBu Optimizer

Two optimizers needed:
1. **Euclidean params** (output weight, norms, biases): standard AdamW
2. **Poincaré params** (embeddings after exp_map, gyration parameters): Riemannian SGD

```python
# Euclidean (AdamW)
m = beta1*m + (1-beta1)*g
v = beta2*v + (1-beta2)*g²
w = w - lr * m/(sqrt(v) + eps) - lr*wd*w

# Hyperbolic (RSGD)  
# Step: w_tangent = log_map(w, R) - lr * g   (in tangent space)
#       w_new = exp_map(w_tangent, R)         (project back to ball)
w_tangent = log_map(w, R)
w_tangent = w_tangent - lr * log_map(g, R)  # map gradient to tangent too
w_new = exp_map(w_tangent, R)
```

**Existing baseline code** has toroidal gradient (`g_wubu = g % 2π`) — this was for a
different math (K-theory torus), not the Poincaré ball. The RSGD above is correct for
hyperbolic space.

## Step 3.4: Training Config

```
batch_size: 2 (fits 6.4GB with 2048 hidden, gradient accumulation to effective 8)
micro_batch_size: 1 (per forward pass)
gradient_accumulation_steps: 4
sequence_length: 4096 (start small, scale to 262K)
learning_rate: 3e-4 (AdamW default)
weight_decay: 0.1
gradient_clip_norm: 1.0 (not 0.1 — 0.1 was for toroidal optimizer)
warmup_steps: 100
lr_schedule: cosine to 3e-5
dtype: float16 (for compute) / float32 (for optimizer states)
```

**Memory budget (6.4GB total):**
```
Model weights (3B active × 2 bytes f16):      ~6GB ← TOO BIG
→ Must use quantization for storage, f16 for compute
→ Strategy: load weights Q5_K/Q8_K, convert to f16 for forward, offload to CPU

KV cache (4096 seq × 40 layers × 2048 × 2B):  ~670MB
Optimizer states (3B × 4 bytes × 2):           ~24GB ← MUST offload to CPU
Activations (2 × 4096 × 2048 × 4B):            ~67MB per layer → ~2.7GB total

Total VRAM needed (with CPU offloading):       ~4GB feasible
```

## Step 3.5: GPU / CUDA Support

Minimum viable CUDA kernels (in priority order):

| Kernel | Priority | Why | Complexity |
|--------|----------|-----|------------|
| MatMul via cuBLAS | 🔴 P0 | 90% of FLOPs | Built-in (cublasSgemm) |
| GQA softmax | 🔴 P0 | 25% of layers | FlashAttention-style tiling |
| SSM scan | 🟡 P1 | Sequential bottleneck | Parallel associative scan |
| MoE grouped matmul | 🟡 P1 | Expert dispatch overhead | Custom grouped GEMM |
| exp/log maps | 🟢 P2 | Element-wise only | Trivial |
| Cross-entropy | 🟢 P2 | Standard | Built-in |

RTX 5050: Ada Lovelace (compute 8.9), 6.4GB VRAM, ~72 TFLOPS f16.
Target: 500+ tok/s training at 4K context.

## Step 3.6: Checkpointing

Save format: GGUF-compatible (write tensors in GGUF format using existing gguf_reader structure)
```
wubu_model.gguf:
  - KV: architecture="wubu", hidden_size=2048, etc.
  - Tensors: token_embd (Q5_K or f16), output_weight (f16 or quantized), 
    blk.N.attn_qkv (Q8_K), blk.N.ssm_* (F32), blk.N.ffn_* (IQ2_XS/IQ1_S)
  - Optimizer state: separate binary (or not saved for inference-only)
```

Frequency: every 1000 steps (save and overwrite to save space).

## Success Criteria
- [x] BBPE tokenizer working (can round-trip tokens ↔ text) ✅ DONE May 13
- [ ] TST superposition phase: bag embeddings + MCE loss (s=8, r=0.25)
- [ ] TST recovery phase: remove TST code, standard CE, loss continues to descend
- [ ] **Parallel associative scan CUDA kernel** (blocking SSM backward pass)
- [ ] SSM backward pass through all 30 layers
- [ ] GQA backward pass through all 10 layers
- [ ] Loss converges below 4.0 on real data (train_stub baseline: 3.466→3.428)
- [ ] GPU training > 10× faster than CPU (>300 tok/s target)
- [ ] Checkpoints save/load correctly in GGUF format
- [ ] Evaluation: perplexity on WikiText-2 evaluation

## Files to Create
```
src/tokenizer.c                 — BBPE tokenizer (Qwen3.5 compatible)
include/tokenizer.h             — Header
src/wubu_model.c                — Full model assemble
include/wubu_model.h            — Header
src/wubu_train.c                — Main training loop
src/adamw.c                     — AdamW optimizer
include/adamw.h                 — Header
src/rsgd.c                      — Riemannian SGD for hyperbolic params
include/rsgd.h                  — Header
Makefile                        — Build system
```

## Pitfalls

1. **BBPE tokenizer is NOT optional.** Without it, the extracted Qwen embeddings cannot
   be used meaningfully. Character-level tokens map to different embedding IDs than BPE tokens.
   The first 200 tokens in BPE are single characters, but token 201+ are multi-character
   subwords. Using token ID 65 (ASCII 'A') maps to embedding for BPE token 65 ("!", depending
   on the merge ordering), which is NOT the embedding for 'A'.

2. **VRAM is the bottleneck.** 3B active params won't fit in f16. Need to:
   - Keep attention/MoE weights in Q8_K/Q5_K, dequant on-the-fly during forward
   - Offload optimizer states to CPU (AdamW states are 2× params = 24GB)
   - Use gradient checkpointing to trade compute for memory

3. **MTP head requires care.** With `mtp_use_dedicated_embeddings=false`, the MTP head
   shares the embedding table. This means the embedding gradient flows through BOTH the
   first-token and second-token prediction paths, which affects training dynamics.

4. **Loss baseline is not directly comparable.** The baseline C model (6 layers, 768 dim,
   97-token char vocab) hits loss 3.12. Our target (40 layers, 2048 dim, 248K vocab)
   will have much higher initial loss because of the larger vocabulary. Target loss should
   be measured against Qwen3.6's actual perplexity, not the char-level baseline.
