# 048 — SFT cold-start corpus: the light DeepSeek/GLM interaction tiers

> Status: `closed` (data fetched + curated). Date: 2026-08-03.
> Maps to: THEME RC03 (RLHF/RLVR), the SFT cold-start step.

## Why this data

The recipe (research/043) says: **SFT cold start ~5-20k examples** on
WuBu-35M (Cosmopedia QA + math CoT + oracle-improved drafts) before any
GRPO/RLVR. The big labs' SFT mixes (DeepSeek, GLM) are built from the
same tiers — ShareGPT/UltraChat-style multi-turn chat, FLAN/Orca-style
general instruction, math CoT — at billion-scale. For a 35M base we take
the **light versions** of exactly those tiers.

## The tiers (light versions)

| Tier | Source | Size | Role in mix |
|---|---|---|---|
| math CoT | openai/gsm8k train | 7,473 | the DeepSeek-math tier (verifiable) |
| chat | HuggingFaceH4/ultrachat_200k | 6,000 sampled | the ShareGPT tier (multi-turn) |
| general | Open-Orca/SlimOrca | 6,000 sampled | the FLAN/Orca tier (instruction) |

**Pack: `/home/wubu/models/corpus/sft/wubu-sft-pack.jsonl` — 19,473
conversations, 48.5 MB.** Format: one JSON per line,
`{"conversations": [{"role": "user"|"assistant", "content": ...}]}`.
Deterministic sample (seed 48) so the pack is reproducible.

## Full downloads (kept for future re-sampling)

`/home/wubu/models/corpus/sft/` (2.5 GB total):
- gsm8k-train/test parquet (2.2 MB + 0.4 MB)
- ultrachat_200k: 3 train_sft + 3 train_gen + test shards (~1.5 GB)
- slimorca-oo-labeled_correct.gpt4.sharegpt.jsonl (985 MB, 517,982 rows)

## Tools (kept in the repo-adjacent tools dir)

- `/home/wubu/research-ponds-work/tools/fetch_sft_data.py` — HF downloader
  (uses HF_READ_TOKEN from ~/.hermes/profiles/mind-palace/secrets/hf.env)
- `/home/wubu/research-ponds-work/tools/build_sft_pack.py` — samples the
  tiers into the deterministic 19,473-conversation pack

## Next step

Tokenize wubu-sft-pack.jsonl with wubu_tokenc.c (conversation role
markers → special tokens) → SFT cold-start training run on the 35M base
→ freeze as π_ref → GRPO/RLVR stage (research/043).

## Storage note

The SD card is COLD storage — the SFT pack is ACTIVE training data, so it
lives on the SSD at /home/wubu/models/corpus/sft/. A cold copy (tar.gz)
belongs on the SD card only when the pack is finalized.
