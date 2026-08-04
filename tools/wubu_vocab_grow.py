#!/usr/bin/env python3
"""wubu_vocab_grow.py -- the amoeba tokenizer GROW operator (v2, correct).

The user's directive (research/054): the tokenizer is the vocab organ —
it must grow toward the corpus. The correct grow (the eBay guarantee:
NEVER more tokens than before) is a mini BPE-training step on top of
the base tokenizer:

  1. Tokenize the domain stream with the BASE tokenizer (wubu_tokenc).
  2. Count adjacent token-pair frequencies in the id stream.
  3. The top-N pairs (not already a merge) become new merges; the new
     token = concatenation of the two sub-token strings.
  4. Init the new embedding rows as the MEAN of the sub-token rows
     (UnifyVocab / eBay Algorithm 2 — deterministic).
  5. The tied head (BL06) reuses the same rows — one knob, both effects.
     Writes the grown tokenizer.json + an embed-init delta sidecar.

Because every new merge only fires when BOTH halves are present, the
token count can never increase — this is the eBay invariant. The
compression GAIN (tokens/doc ↓) is the diagnose metric.

Usage:
  python3 tools/wubu_vocab_grow.py \
      --tokenizer models/wubu/tokenizer.json \
      --corpus /home/wubu/models/corpus/sft-text/wubu-agentic.txt \
      --top 512 --out models/wubu/tokenizer.json.grown
"""
import argparse
import json
import subprocess
import sys
from collections import Counter


def tokenize(base_tokenc, tokenizer_path, corpus_path, out_tok):
    r = subprocess.run([base_tokenc, tokenizer_path, corpus_path, out_tok],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        sys.exit(f"tokenc failed: {r.stderr}")
    import struct
    data = open(out_tok, "rb").read()
    return struct.unpack("<%dH" % (len(data) // 2), data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokenc", default="/home/wubu/wubuwizard/wubu_tokenc")
    ap.add_argument("--top", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = json.load(open(a.tokenizer, encoding="utf-8"))
    vocab = tok["model"]["vocab"]      # {token_str: id}
    merges = tok["model"]["merges"]    # [[l, r], ...]
    id_to_str = {v: k for k, v in vocab.items()}
    merge_set = {tuple(m) for m in merges}
    next_id = max(vocab.values()) + 1

    # 1. tokenize the domain stream with the base tokenizer
    ids = tokenize(a.tokenc, a.tokenizer, a.corpus, "/tmp/vg_toks.tok")
    print(f"grow: base tokenizer -> {len(ids)} tokens")

    # 2. count adjacent token pairs (skip <bos>/<eos> boundaries ~ docs)
    pairs = Counter()
    for i in range(len(ids) - 1):
        l, r = ids[i], ids[i + 1]
        # never merge across a special token (2=bos, 3=eos, 4-6 roles)
        if l < 7 or r < 7:
            continue
        pairs[(l, r)] += 1

    # 3. top pairs not already merges -> new tokens
    added = 0
    delta = {}   # new_id -> [l_str, r_str]
    additions = []  # (new_str, [l_str, r_str])
    for (l, r), cnt in pairs.most_common():
        if added >= a.top:
            break
        l_str, r_str = id_to_str[l], id_to_str[r]
        if (l_str, r_str) in merge_set:
            continue
        new_str = l_str + r_str
        if new_str in vocab:
            continue
        additions.append((new_str, [l_str, r_str], cnt))
        merge_set.add((l_str, r_str))
        added += 1

    for new_str, (l_str, r_str), cnt in additions:
        vocab[new_str] = next_id
        merges.insert(0, [l_str, r_str])   # front = highest priority
        delta[str(next_id)] = [l_str, r_str]
        next_id += 1

    print(f"grow: added {added} tokens (vocab {len(vocab)-added} -> {len(vocab)})")
    tok["model"]["vocab"] = vocab
    tok["model"]["merges"] = merges
    json.dump(tok, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    with open(a.out + ".delta.json", "w", encoding="utf-8") as f:
        json.dump({k: [vocab[l], vocab[r]] for k, (l, r) in delta.items()},
                  f, ensure_ascii=False)
    print(f"wrote {a.out} + {a.out}.delta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
