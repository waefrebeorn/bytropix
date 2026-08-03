#!/usr/bin/env python3
"""
barun_extract.py -- extract text from the parquet shards on the SD card.

The C11 tokenizer (barun_tokenc) consumes plain text, not parquet.
This tool streams each parquet shard to .txt (one doc per paragraph-ish
line), then the C tokenizer converts to .tok.

Usage:
    python3 barun_extract.py cosmopedia-v2 0 0     # -> corpus/text/<shard>.txt
"""
import os, sys, json

import pyarrow.parquet as pq

SD = "/home/wubu/sdcard/corpus"
RAW = os.path.join(SD, "raw")
TXT = os.path.join(SD, "text")
LOG = os.path.join(SD, "logs")
os.makedirs(TXT, exist_ok=True)

def extract(name, lo, hi):
    for i in range(lo, min(hi, 10000) + 1):
        src = os.path.join(RAW, f"{name}-{i:05d}.parquet")
        dst = os.path.join(TXT, f"{name}-{i:05d}.txt")
        if not os.path.exists(src):
            print(f"  missing {src}")
            continue
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            print(f"  skip {dst}")
            continue
        print(f"  extract {name} shard {i} ...", flush=True)
        table = pq.read_table(src)
        col = None
        for c in table.column_names:
            if c in ("text", "content", "document"):
                col = c; break
        if col is None: col = table.column_names[0]
        n_docs = 0
        with open(dst, "w") as f:
            for batch in table.to_batches():
                for doc in batch.column(col).to_pylist():
                    if not doc or not isinstance(doc, str): continue
                    # normalize whitespace, one paragraph per line
                    paras = [p.strip().replace("\n", " ") for p in doc.split("\n\n")]
                    for p in paras:
                        if len(p) > 20: f.write(p + "\n")
                    f.write("\n")   # blank line = document boundary
                    n_docs += 1
        print(f"    -> {n_docs} docs, {os.path.getsize(dst)//1024//1024} MB")
        with open(os.path.join(LOG, f"{name}-{i:05d}-extract.json"), "w") as f:
            json.dump({"docs": n_docs, "bytes": os.path.getsize(dst)}, f)

if __name__ == "__main__":
    extract(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
