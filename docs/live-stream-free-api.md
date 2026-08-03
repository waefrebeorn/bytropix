# Live-Stream + Free API Integration — 2026-08-03

## What works (verified at runtime)

### NVIDIA NIM free tier (build.nvidia.com) — VERIFIED LIVE
- Endpoint: `https://integrate.api.nvidia.com/v1` (OpenAI-compatible)
- Key: 3x `nvapi-*` stored in `~/.hermes/profiles/mind-palace/secrets/hf.env` (0600)
- **Embeddings** (live-stream embedding): `nvidia/nv-embed-v1` → 4096 dims ✓,
  `nvidia/nemotron-3-embed-1b` → 2048 dims ✓
- **Chat** (the oracle): `minimaxai/minimax-m3` ✓, `z-ai/glm-5.2` ✓,
  `deepseek-ai/deepseek-v4-pro` ✓ (102 models listed; many 404/410 = deprecated
  names, the three above are the verified-live ones)
- **RLHF oracle** (`score_draft`): scored WuBu's hive draft 35/100 with a
  critique — the R1/Prover reward loop, live, free
- Client: `tools/nvidia_nim.py` (embed + chat + score_draft)

### OpenRouter free tier — CLIENT BUILT, QUOTA-LIMITED NOW
- Endpoint: `https://openrouter.ai/api/v1`
- 6x `sk-or-v1-*` keys (round-robin in the client) for RLHF
- 337 models, 14 `:free` (gemma-4-31b-it, nemotron-3-super/ultra-550b, ling-3.0)
- **All 6 keys returned 429 (rate-limited) at first use** — the :free models
  are congested; the keys are valid (the models endpoint authenticated fine).
  Retry later; the client is ready.
- Client: `tools/openrouter_rlhf.py` (rlhf_improve: score + critique + improved)

## Live-stream data pipeline (`tools/wubu_stream.py`)
- Streams HF datasets row-by-row (parquet files, one at a time — no full download)
- Tokenizes with our own BPE vocab (byte-level fallback in Python; the exact
  C11 tokenizer stays the reference)
- Verified: finemath 2000 docs → 153K tokens at ~3000 tok/s
- Backgrounded: finemath-live (full) + openmath-live (full) → SD card tokens/
- Best-dataset map (all accessible with our read token):
  - HuggingFaceTB/finemath 17.8GB (DeepSeekMath lineage — our Lean vault)
  - nvidia/OpenMathReasoning 8.2GB (NVIDIA reasoning)
  - HuggingFaceFW/fineweb-edu 88.2GB (education)
  - HuggingFaceTB/smollm-corpus (already tokenizing: shard 0 = 135.6M tokens)

## Design notes
- The NVIDIA oracle IS the live RLHF signal: WuBu drafts → frontier scores →
  reward → the trainer consumes it. This is the bigger-brother doctrine
  without the brother's weights — the brother now lives at an API endpoint.
- The 6 OpenRouter keys give redundancy when the NVIDIA credits run out
  (~1000 credits, one credit ≈ one call).
- All keys are in secrets/ (0600), never in git.
