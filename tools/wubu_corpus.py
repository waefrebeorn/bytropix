#!/usr/bin/env python3
"""
wubu_corpus.py -- the AGI corpus acquisition + tokenization pipeline.

Downloads the smollm-corpus shards (the exact corpus BarunLM was trained
on) to the SD card, extracts the text, tokenizes with the byte-level BPE
tokenizer, and writes compact uint16 token streams (.tok) that the C11
trainer consumes.

Layout on the SD card (D:, /home/wubu/sdcard):
    corpus/raw/<name>-<shard>.parquet     downloaded shards
    corpus/tokens/<name>-<shard>.tok      token streams (uint16 LE)
    corpus/logs/                           acquisition ledger

Usage:
    python3 wubu_corpus.py list          # what is available
    python3 wubu_corpus.py fetch 0 4     # download shards 0..4 (cosmopedia-v2)
    python3 wubu_corpus.py tokenize 0 4  # tokenize shards 0..4
"""
import json, os, struct, sys, urllib.request, hashlib

BASE = "https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/main"
SD = "/home/wubu/sdcard/corpus"
RAW = os.path.join(SD, "raw")
TOK = os.path.join(SD, "tokens")
LOG = os.path.join(SD, "logs")
os.makedirs(RAW, exist_ok=True)
os.makedirs(TOK, exist_ok=True)
os.makedirs(LOG, exist_ok=True)

SHARDS = {
    "cosmopedia-v2": 104,
    "fineweb-edu": 103,
    "finemath-4plus": 8,
}

def shard_url(name, i):
    return f"{BASE}/{name}/train-{i:05d}-of-{SHARDS[name]:05d}.parquet"

def fetch(name, lo, hi):
    for i in range(lo, min(hi, SHARDS[name]) + 1):
        dst = os.path.join(RAW, f"{name}-{i:05d}.parquet")
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            print(f"  skip {dst} (exists)")
            continue
        url = shard_url(name, i)
        print(f"  fetch {name} shard {i} ...", flush=True)
        try:
            urllib.request.urlretrieve(url, dst)
            print(f"    -> {os.path.getsize(dst)//1024//1024} MB")
        except Exception as e:
            print(f"    FAIL {e}")
            break

def tokenize(name, lo, hi):
    import pyarrow.parquet as pq
    from pathlib import Path
    # load the BPE tokenizer (byte-level, our release tokenizer)
    tok_data = json.load(open("models/wubu/tokenizer.json"))
    vocab = tok_data["model"]["vocab"]
    merges = tok_data["model"]["merges"]
    # byte-level pre-tokenizer tables
    byte_to_bl = {}
    # build the byte->unicode mapping the HF byte-level BPE uses
    n = 0
    bs = sorted(set(range(256)) - set(range(0x21, 0x7F)))
    for b in range(256):
        if 0x21 <= b < 0x7F:
            byte_to_bl[b] = chr(b)
        else:
            byte_to_bl[b] = chr(256 + n); n += 1

    def encode_text(text):
        # byte-level split
        syms = [byte_to_bl[b] for b in text.encode("utf-8")]
        # apply merges (rank order)
        merge_rank = {tuple(m) if isinstance(m, list) else tuple(m.split()): idx
                      for idx, m in enumerate(merges)}
        changed = True
        while changed:
            changed = False
            best_rank = None; best_i = None
            for i in range(len(syms) - 1):
                pair = (syms[i], syms[i+1])
                if pair in merge_rank:
                    r = merge_rank[pair]
                    if best_rank is None or r < best_rank:
                        best_rank = r; best_i = i
            if best_i is not None:
                syms[best_i:best_i+2] = [syms[best_i] + syms[best_i+1]]
                changed = True
        return [vocab[s] for s in syms if s in vocab]

    for i in range(lo, min(hi, SHARDS[name]) + 1):
        src = os.path.join(RAW, f"{name}-{i:05d}.parquet")
        dst = os.path.join(TOK, f"{name}-{i:05d}.tok")
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            print(f"  skip tokenize {dst} (exists)")
            continue
        if not os.path.exists(src):
            print(f"  missing {src} -- fetch first")
            continue
        print(f"  tokenize {name} shard {i} ...", flush=True)
        table = pq.read_table(src)
        text_col = None
        for col in table.column_names:
            if col in ("text", "content", "document"):
                text_col = col; break
        if text_col is None:
            text_col = table.column_names[0]
        n_tokens = 0; n_docs = 0
        out = bytearray()
        for batch in table.to_batches():
            for doc in batch.column(text_col).to_pylist():
                if not doc or not isinstance(doc, str): continue
                ids = encode_text(doc)
                if not ids: continue
                n_docs += 1
                # <bos> doc <eos>
                for t in ([2] + ids + [3]):
                    out += struct.pack("<H", t)
                n_tokens += len(ids) + 2
        with open(dst, "wb") as f:
            f.write(out)
        print(f"    -> {n_docs} docs, {n_tokens} tokens, {len(out)//1024//1024} MB")
        with open(os.path.join(LOG, f"{name}-{i:05d}.json"), "w") as f:
            json.dump({"shard": i, "docs": n_docs, "tokens": n_tokens}, f)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for name, n in SHARDS.items():
            have = len([f for f in os.listdir(RAW) if f.startswith(name)])
            print(f"{name}: {have}/{n} shards fetched")
        for f in sorted(os.listdir(TOK))[:5]:
            print(f"  token: {f} {os.path.getsize(os.path.join(TOK,f))//1024} KB")
    elif cmd == "fetch":
        fetch(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "tokenize":
        tokenize(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        print("usage: wubu_corpus.py list|fetch <name> <lo> <hi>|tokenize <name> <lo> <hi>")
