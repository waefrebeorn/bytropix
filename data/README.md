# wubuwizard/data/ — tokenizer + corpus metadata

This directory holds the tokenizer vocabulary/merges and small corpus
metadata. The BIG training data lives OUTSIDE the repo:

- `/home/wubu/models/corpus/` — active training corpus (tokens, SFT pack,
  agentic pack). Master manifest: `CORPUS.md` there.
- `/home/wubu/sdcard/corpus/` — cold raw archive (SD card).
- `/home/wubu/sdcard/archive/` — cold storage for large artifacts
  (e.g. qwen36_embeddings_c.bin.raw.tar.gz, research ponds).

## Files

| File | What it is |
|---|---|
| vocab.bin / merges.bin | Compiled tokenizer tables (C11 loader format) |
| special_tokens.bin | Special-token ids |
| tokenizer_vocab.txt / tokenizer_merges.txt | Human-readable tokenizer dump |
| tokenizer_vocab_raw.txt / tokenizer_merges_raw.txt | Pre-dedup raw dump |
| corpus_raw.txt | Small seed corpus (4.4 MB) for smoke tests |
| dataset_stats.json | Corpus statistics |
| train_meta.txt | Training metadata |
| prepare_data.py | Corpus prep script |
| moondream3_vision_config.txt / moondream3_vision_index.json | Vision-side config |
| qwen36_embeddings_c.bin.raw | REMOVED 2026-08-04 — 2 GB transfer artifact, cold-archived to SD (/home/wubu/sdcard/archive/qwen36_embeddings_c.bin.raw.tar.gz). Loaded only if present AND vocab matches (src/wubu_model.c:505,1615). Optional. |

## Notes

- `.gitignore` excludes `*.bin.raw` — large binary transfer artifacts
  never enter the repo.
- The tokenizer is our C11 byte-level BPE (vocab 16384); see
  `tools/wubu_tokenc.c`.
