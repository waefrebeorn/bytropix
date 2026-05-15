# WuBuText AI — Project Overview (May 16 v7 — HONEST)

## Mission
Build Qwen3.6-35B-A3B from scratch in pure C + CUDA with WuBu nested hyperbolic geometry.
**HARD TRUTH: Inference is BROKEN.** All inference binaries produce garbage output.
Everything depends on fixing this first.

## All Phases — HONEST Status

| Phase | Component | Claimed | Real Status |
|-------|-----------|---------|-------------|
| 0 | GGUF Tensor Layout | ✅ | ✅ Works — 733 tensors, 13 types |
| 1 | Embedding Graft | ✅ 95% NN | ❓ 95% NN claim unverified against raw embeddings |
| 2 | Attention (SSM+GQA) | ✅ | ❌ Produces wrong output in inference |
| 3 | Training Loop | ✅ 11s/step | ❓ CE measured but no reference baseline |
| 4 | MoE Port | ✅ 256 exp | ❌ MOE=1 also produces garbage |
| 5 | Vision Port | ✅ 99ms | ❓ Moondream3 model — separate from Qwen |
| 6 | CUDA Kernels | ✅ | ✅ max_diff<6e-8 for SSM + MoE dispatch |

**Only 2/8 binaries verified correct:** `test_kv_cache`, `test_256k`.

## What Actually Works
- **test_kv_cache**: KV cache matches full recompute (max_diff=0.00) ✅
- **test_256k**: MoE router O(T) scaling to 65K ✅
- **API server**: tools/serve.py sandbox mode (14 tests pass) ✅
- **llama.cpp reference**: BUILT at ~/llama.cpp/build/bin/llama-cli ✅
- **Individual components**: SSM, GQA, MoE forward passes compile & run ✅
- **SGEMM ldC bug**: FIXED (was all-zero logits)
- **RoPE**: Added to CPU GQA prefill path
- **Sampling**: temp/top-k/top-p added
- **EOS detection**: Fixed (eos=bos=248044)
- **NaN in training**: FIXED (MoE weight interleaving root cause)

## What's Broken (P0 — Fix First)
- **ALL inference binaries produce garbage** (infer_text_gpu, infer_text, MOE=1)
- Root cause unknown: SSM impl? MoE dequant? Tokenizer?
- 6/15 math components forward-only (no gradient flow for backward)
- Q5_K dequant fix may be wrong (old and new code produce same values)

## Key Achievements (Still Real)
- gguf_raw_size(IQ2_XXS) fix: 72→66 bytes/block — eliminated NaN cascade
- Per-expert IQ2_XXS dequant: 3.9ms/expert — 177s→11s/step (16×)
- GPU output projection: cublasSgemm replaces 2B CPU FMAs
- 6 env flags all verified individually + combined, 0 NaN

## Remaining

| Issue | Severity |
|-------|----------|
| Inference produces garbage (P0 — ALL EFFORT HERE) | **CRITICAL** |
| ~11s/step GPU compute bound | Performance |
| PGA loss jump (21.6→69) | Numeric |
| CONV_DIM=8192 vs config 1536 | Possible bug |
| MRoPE 3D not implemented | Correctness |
| MTP prediction head missing | Feature |
| 12 vaults with unported theory | P2 |
| Tailslayer spec-decode kernel | P2 |
