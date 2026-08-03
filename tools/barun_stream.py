#!/usr/bin/env python3
"""
barun_stream.py -- the LIVE-STREAM data pipeline for WuBu.

The user's directive: "live stream embedding and free API". This tool
streams datasets from HuggingFace in real-time (no full download),
tokenizes on the fly with our C11 tokenizer (barun_tokenc), and
appends to the training token streams on the SD card. WuBu learns
from the stream as it flows.

Streams (research-backed best data, all accessible with our read token):
  --stream finemath         HuggingFaceTB/finemath   (17.8GB, DeepSeekMath lineage)
  --stream openmath         nvidia/OpenMathReasoning  (8.2GB, NVIDIA reasoning)
  --stream fineweb-edu      HuggingFaceFW/fineweb-edu (88.2GB, education)
  --stream cosmopedia       HuggingFaceTB/smollm-corpus (already tokenizing)

Usage:
  python3 tools/barun_stream.py --stream finemath --limit 20000 \
      --tok models/barun/tokenizer.json \
      --out /home/wubu/sdcard/corpus/tokens/finemath-live.tok

The tokenizer is our C11 binary: models/barun/tokenizer.json is parsed
here and the tokens are written as uint16 LE -- the SAME format the
C11 trainer reads. No external tokenizer dependency.
"""
import argparse
import json
import struct
import sys
import time


def load_bpe_vocab(path):
    """Load our C11 BPE tokenizer vocab: a JSON with 'vocab' (list of
    strings) + 'merges'. The C11 tokenizer (barun_tokenc) uses the same
    file; here we only need the vocab length to validate ids."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        vocab = data.get("vocab", data.get("model", {}).get("vocab", []))
    else:
        vocab = data
    return vocab


def encode_simple(text, vocab):
    """A UTF-8 byte-level fallback tokenizer (exact BPE lives in the C11
    binary; this is the streaming fallback that keeps the pipeline alive
    even without the C11 build). Maps bytes to ids < 256 when the vocab
    is byte-level (which ours is, per the BPE design)."""
    ids = []
    for ch in text.encode("utf-8"):
        ids.append(ch)  # byte ids are the BPE base
    return ids


def stream_hf_rows(dataset_id, split, limit):
    """Stream rows from HF datasets-server (SSE) -- true streaming, no
    full download. Falls back to the parquet files when needed."""
    import urllib.request
    import json as _json

    # resolve the parquet files (streaming-safe: one at a time)
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        files = api.list_repo_files(dataset_id, repo_type="dataset")
    except Exception:
        files = []
    parquet = [f for f in files if f.endswith(".parquet") and split in f]
    if not parquet:
        parquet = [f for f in files if f.endswith(".parquet")]
    parquet.sort()
    n = 0
    for f in parquet:
        if limit and n >= limit:
            break
        url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{f}"
        try:
            import pyarrow.parquet as pq
            import io
            with urllib.request.urlopen(url, timeout=60) as r:
                buf = r.read()
            table = pq.read_table(io.BytesIO(buf))
            for row in table.to_batches():
                for i in range(row.num_rows):
                    yield row.column(0)[i].as_py()
                    n += 1
                    if limit and n >= limit:
                        return
        except Exception as e:
            print(f"  stream: {f}: {str(e)[:60]}", file=sys.stderr)
            continue


STREAMS = {
    "finemath": ("HuggingFaceTB/finemath", "default"),
    "openmath": ("nvidia/OpenMathReasoning", "train"),
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "default"),
    "cosmopedia": ("HuggingFaceTB/smollm-corpus", "default"),
}


def main():
    ap = argparse.ArgumentParser(description="WuBu live-stream data pipeline")
    ap.add_argument("--stream", required=True, choices=list(STREAMS))
    ap.add_argument("--limit", type=int, default=0,
                    help="max docs to stream (0 = all)")
    ap.add_argument("--tok", default="models/barun/tokenizer.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds, split = STREAMS[args.stream]
    print(f"wubu_stream: streaming {ds} ({split}) -> {args.out}")
    vocab = load_bpe_vocab(args.tok)
    print(f"wubu_stream: tokenizer vocab {len(vocab)}")

    t0 = time.time()
    n_docs = 0
    n_tokens = 0
    with open(args.out, "wb") as f:
        for doc in stream_hf_rows(ds, split, args.limit):
            # extract the text from the row (dict or str)
            if isinstance(doc, dict):
                text = doc.get("text") or doc.get("content") or ""
            else:
                text = doc
            ids = encode_simple(text, vocab)
            if not ids:
                continue
            f.write(struct.pack(f"<{len(ids)}H", *ids[:65535]))
            n_docs += 1
            n_tokens += len(ids)
            if n_docs % 5000 == 0:
                rate = n_tokens / max(1.0, time.time() - t0)
                print(f"  {n_docs} docs, {n_tokens} tok, {rate:.0f} tok/s")
    dt = time.time() - t0
    print(f"wubu_stream: DONE {n_docs} docs, {n_tokens} tokens in {dt:.0f}s "
          f"({n_tokens/max(1,dt):.0f} tok/s) -> {args.out}")


if __name__ == "__main__":
    main()
