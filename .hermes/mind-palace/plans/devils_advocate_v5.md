# Devil's Advocate v5 — Meta Audit: All Goals, Math, Models

## Date: May 16, 2026
## Purpose: Honest audit of every claim, every binary, every math formula. No survivorship bias.

---

## GROUND TRUTH: Does The Inference Work?

**Run:** `TEMP=0.0 MOE=1 ./infer_text "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf" "The capital of France is" 15`
**Our output:** `<|endoftext|>The capital of France isiscInset了下去idesiby客的我们都会论usher...волоatism οικονyleconf`
**llama.cpp output:** `Here's a thinking process:\n\n1.`

**Verdict: OUR INFERENCE IS BROKEN. Q5_K dequant FIXED and TGT wrap REMOVED this session, but root cause remains. Output changed but still garbage.**

State.md lists 8 inference binaries as ✅. NONE have been verified to produce correct output against a reference.

---

## All Binaries: Real Status

| Binary | Claimed | Actual | Evidence |
|--------|---------|--------|----------|
| `infer_text_gpu` | ✅ 245 tok/s | ❌ Garbage output | "The capital of France is  ิ" |
| `infer_text` v2 | ✅ | ❌ Garbage output | " The capital of France isiscInset..." |
| `train_integrated` | ✅ CE 21.6→18.4 | ❓ CE measured but no reference to compare against | Loss goes down but could be overfitting garbage |
| `infer_moe_lazy` | ✅ 37 tok/s | ❓ Speed claim but output never verified | Dequant benchmark ≠ inference |
| `infer_unified` | ✅ | ❌ Same architecture, same bugs | 40 layers but same broken SSM/GQA |
| `test_kv_cache` | ✅ max_diff=0.00 | ✅ Cache vs recompute matches | Numerical comparison valid |
| `test_256k` | ✅ MoE router 65K | ✅ Individual component test | But only MoE router, not end-to-end |
| `infer_vision_gpu` | ✅ 99ms | ❓ Vision model (Moondream3) is separate from Qwen3.6 | Different model entirely |

**Only 2/8 binaries have verified correctness: `test_kv_cache` and `test_256k`. All inference binaries are unverified.**

---

## All Math: What's Actually Used vs What's Dead Code

| Math Component | Used In | Actually Works? | Can Verify? |
|---------------|---------|----------------|-------------|
| **Standard GQA attention** | infer_text, infer_text_gpu | ❌ Produces wrong output | Compare hidden states vs llama.cpp |
| **Gated DeltaNet (SSM)** | infer_text, infer_text_gpu | ❌ Same pipeline, same wrong output | Compare SSM recurrence vs llama.cpp |
| **SwiGLU activation** (MoE) | infer_text MOE=1 | ❌ MOE=1 also wrong | Expert dequant may be buggy |
| **RMSNorm** | All layers | ✅ Individual unit test passes | wubu_rms_norm verified |
| **RoPE** | GQA layers | ❓ Added to CPU path but output unchanged | May be applied incorrectly |
| **TGT (Trigonometric Gradient Trick)** | train_integrated | ❓ Individual test passes | Used in training only |
| **Poincaré ball (hyperbolic)** | Poincaré GQA, embedding graft | ❓ Individual tests pass | Forward only, backward missing |
| **Möbius addition** | Poincaré SSM, hyperbolic | ❓ src/wubu_mobius.c has tests | Forward verified, backward not |
| **RSGD (Riemannian SGD)** | train_integrated RSGD=1 | ❓ Works for 1D, 2D case | Full-scale gradient projection unknown |
| **Nested SSM (K=4)** | NESTED_SSM=1 in training | ❓ Forward only, backward missing | 3/3 tests pass for forward |
| **Nested MoE** | NESTED_MOE=1 | ❓ Forward only | 396/396 tests for forward |
| **TST (Token Superposition Training)** | TST=1 | ❓ 8/8 tests pass | Bag+MCE loss verified at small scale |
| **PGA (Poincaré GQA Attention)** | PGA=1 | ❌ Forward only, backward missing | No gradient flow for hyperbolic attn |
| **IQ2_XXS/IQ3_XXS/Q5_K/Q6_K dequant** | All weight loading | ⚠️ Q5_K dequant bug **FIXED** (qh indexing matched). Q4_K verified. Others unknown | Need cross-reference vs llama.cpp for Q6_K, IQ2_XXS, IQ2_S |
| **BPE tokenizer** | All inference | ❓ Custom tokenizer from pre-extracted files | No comparison with llama.cpp's GGUF-native tokenizer |

---

## All Models: Relationships

| Model | File | Role | Status |
|-------|------|------|--------|
| **Qwen3.6-35B-A3B** | `models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf` | Base model (all inference targets this) | Production model, 11GB GGUF |
| **Embeddings (Qwen3.6)** | `data/qwen36_embeddings_c.bin` | Pre-extracted token embeddings | 2GB binary, extracted from GGUF |
| **Moondream3 vision** | `data/moondream3_vision_weights.bin` | Vision encoder (separate model) | 1.7GB, Moondream3 VL model |
| **Training data** | `data/train_data.bin` | Custom training corpus | 4MB, openwebmath sample |
| **Custom training model** | Not saved as separate binary | Trainable params via `train_integrated` | 0.9B params per expert, hyperbolic embeddings |

---

## The Real Priority Queue

### P0 — Fix Production Inference
Without this, ALL other work (hyperbolic, training, TST, vision) is built on a broken foundation.
- Compare hidden states layer-by-layer against llama.cpp
- Fix SSM forward pass (most likely root cause)
- Fix MoE expert dequant (IQ2_XXS, IQ3_XXS)
- Fix tokenizer (use GGUF-native, not pre-extracted)
- After fix: `TEMP=0.0 MOE=1` must produce "Here's a thinking process:" not garbage

### P1 — Verify All Already-Existing Components
- Train_integrated: does CE actually correspond to real learning? Compare against known loss values.
- Poincaré graft: 95% NN preservation claimed — verify against raw Euclidean embeddings
- All benchmarks: Rerun with correct inference, not garbage-in/garbage-out

### P2 — Hyperbolic Training (Poincaré GQA backward, Nested SSM backward)
- Currently forward-only: can't train hyperbolic attention or nested components
- Without backward passes, hyperbolic training is aspirational

### P3 — GPU Acceleration & Scaling
- MoE on-device dequant
- Tailslayer spec decode
- 256K context verification

---

## Honest Accounting: What We Have vs What We Claim

| Claim | Reality |
|-------|---------|
| "245 tok/s decode" | Speed is real but output is garbage, making the speed meaningless |
| "40-layer GPU inference" | 40 layers run, but produce wrong output |
| "Chunked prefill + KV cache verified" | Architecture works numerically (test_kv_cache) but full pipeline broken |
| "MoE lazy dequant 9x faster" | Dequant speed is real, but MoE output is wrong |
| "SSM state carry verified" | State carry works numerically, but SSM output is wrong |
| "All 12 training streams integrated" | All wired, but most are forward-only — no gradient flow |
| "CE loss 12.42" | Measured, but no reference to compare against (random baseline?) |
| "95% NN preservation (Poincaré)" | Individual test passes, but never verified against real Euclidean embeddings |
| "API server built" | ✅ LEGIT — works in sandbox mode. Real inference subprocess still produces garbage |

---

## What We Should Do

1. **Fix inference first.** Everything depends on it. Use llama.cpp as ground truth.
2. **Audit every binary.** Remove ✅ status from binaries that haven't been verified against reference.
3. **Golden outputs for everything.** Not just "life!!!" — real instruction-following completions.
4. **Forward-only training is not training.** Backward passes for hyperbolic components must be written before claiming integration.
5. **Vault old mind palace versions** instead of overwriting. Store v9, v10, v11 as records in vault/bins/.

---

## Vault Versioning

Current vault state is ephemeral — files overwritten each session.
Proposal: Archive mind palace snapshots to `vault/bins/YYYY-MM-DD-v{N}/` before each major rewrite.
This gives us audit trail: what did we think was true on May 15 vs May 16.

---

## Session Progress (May 16 PM)

### Fixed This Session
1. **Q5_K dequant qh bit-indexing** — was treating qh as linear 32-byte bitfield. Reference stores 4 interleaved pairs per byte. **Fixed** — output changed but still garbage.
2. **TGT state wrap removed** — SSM forward no longer wraps state to [-π,π] each step. **No effect** on 6-token sequences.
3. **GQA gate verified** — sigmoid(gate) × attn_out applied correctly in both prefill + decode.
4. **EOS detection verified** — gen>1 check correct for eos=bos=248044.

### Key Diagnostics
- **Embedding mean=0.02** (normal), **Layer 0 SSM out mean=2.85** (140× larger — suspicious)
- **llama.cpp reference confirmed**: "Here's a thinking process" for "The capital of France is"
- **MOE=0**: loops `ò`(21502)/`_tuples`(86196) — stuck in 2-token attractor
- **MOE=1**: jumps between languages each step — different token each time but semantically dead

### Remaining Suspects
1. **SSM output projection weight dequant**: `ssm_out.weight` Q5_K dequant may still produce wrong float values
2. **Other dequant types**: Q6_K, IQ2_XXS, IQ2_S weights not verified vs reference
3. **SSM recurrence formula**: Gated DeltaNet implementation may differ from Qwen3.6
4. **Tokenizer**: Custom tokenizer may produce different token IDs than GGUF-native
