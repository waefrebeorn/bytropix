# `data/` — Pre-extracted Embeddings, Training Data, and Reference Vectors

| File | Purpose |
|------|---------|
| `qwen36_embeddings.bin.meta` | Pre-extracted token embedding table (248320×2048, Q5_K) |
| `merges.bin` | BPE merge table (247587 merges) for tokenizer |
| `gqa_test_vectors.bin` | Test vectors for GQA attention verification |
| `moondream3_vision_config.txt` | Vision encoder config (extracted from vLLM) |
| `moondream3_vision_index.json` | Vision weight tensor index |
| `prepare_data.py` | Training data preparation script |
| `corpus_raw.txt` | Raw training corpus sample |
| `openwebmath_sample.jsonl.gz` | OpenWebMath training sample |
| `dataset_stats.json` | Dataset statistics |

## Embedding File

The token embedding table is pre-extracted from the GGUF into a flat binary for faster loading:
- Shape: [248320, 2048] = 1.9 GB as F32
- Loaded at init from `data/qwen36_embeddings_c.bin.raw`
- Falls back to live GGUF extraction if file missing
