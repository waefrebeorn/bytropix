#!/usr/bin/env python3
"""gather_micro_pretrain.py -- fetch raw-text micro-pretraining corpora.

The SFT/eval sets in models/corpus/micro are not pretraining data. This tool
pulls *raw text* at a small, high-quality scale, under an explicit byte budget
and licence gate, into models/corpus/micro_pretrain/<name>/.

Design rules:
  - public datasets only, licence recorded per shard, no credentials;
  - a hard --budget (default 400 MB) stops mid-dataset, never overruns;
  - eval-only and non-commercial sources are refused, not silently included;
  - every shard gets a MANIFEST.json (id, licence, bytes, sha256, rows).

Usage:
  tools/gather_micro_pretrain.py --list
  tools/gather_micro_pretrain.py --budget-mb 200 --only tinystories,cosmopedia
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "models", "corpus", "micro_pretrain")

# (name, hf_id, config, split, text_field, licence, category, why)
CATALOG = [
    ("tinystories", "roneneldan/TinyStories", None, "train", "text", "cdla-sharing-1.0",
     "narrative", "tiny vocab, clean syntax: the cheapest way to teach grammar"),
    ("cosmopedia", "HuggingFaceTB/cosmopedia", "web_v1", "train", "text", "apache-2.0",
     "general", "synthetic textbook prose matched to small-model curricula"),
    ("finemath", "HuggingFaceTB/finemath", "finemath-3plus", "train", "text", "apache-2.0",
     "math", "filtered educational math, dense reasoning per token"),
    ("fineweb-edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", "odc-by",
     "general", "the reference high-quality web filter; sample only"),
    ("the-stack-smol", "bigcode/the-stack-smol", None, "train", "content", "permissive",
     "code", "permissively licensed code, small enough to hold"),
    ("wikipedia-simple", "wikipedia", "20231101.simple", "train", "text", "cc-by-sa-4.0",
     "knowledge", "simple English: short sentences, factual density"),
]

REFUSED = {
    # id: reason -- keeps the gate honest instead of quietly skipping
    "Jev-RAG": "CC BY-NC: research-only, never a training shard",
}


def human(n):
    return f"{n/1e6:.1f} MB"


RAW = {
    # name: (url, delimiter) -- direct files; no rate-limited rows API
    "tinystories": ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
                    "<|endoftext|>"),
}


def fetch_raw(name, url, delim, licence, category, why, budget_left, dry_run):
    """Stream the first `budget_left` bytes of a raw text file as JSONL lines."""
    import urllib.request
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    path = os.path.join(OUT, name, "shard-000.jsonl")
    part = path + ".part"
    if dry_run:
        print(f"  [dry] {name}: would stream {human(budget_left)} from {url}")
        return 0, 0, None
    written = rows = 0
    h = hashlib.sha256()
    got = b""
    try:
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{budget_left-1}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            got = r.read()
    except Exception as e:
        print(f"  [warn] {name}: {type(e).__name__}: {e}", file=sys.stderr)
        return 0, 0, None
    with open(part, "wb") as f:
        for chunk in got.split(delim.encode()):
            text = chunk.strip()
            if len(text) < 64:
                continue
            line = json.dumps({"text": text.decode("utf-8", "replace")},
                              ensure_ascii=False).encode() + b"\n"
            f.write(line); h.update(line)
            written += len(line); rows += 1
            if written >= budget_left:
                break
    if rows == 0:
        os.unlink(part); return 0, 0, None
    os.replace(part, path)
    man = {"id": url, "licence": licence, "category": category, "why": why,
           "bytes": written, "rows": rows, "sha256": h.hexdigest(),
           "source": "direct file (HTTP range)"}
    with open(os.path.join(OUT, name, "MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=1)
    return written, rows, man


def fetch_one(name, hf_id, config, split, field, licence, category, why,
              budget_left, dry_run):
    """Stream rows from the HF datasets-server and write one JSONL shard.

    Uses the public rows endpoint so no `datasets` dependency and no token.
    """
    import urllib.request

    if not config:
        # The rows endpoint requires a config; ask the server for the first one.
        try:
            u = ("https://datasets-server.huggingface.co/splits?dataset="
                 + urllib.parse.quote(hf_id))
            with urllib.request.urlopen(u, timeout=30) as r:
                sp = json.load(r).get("splits") or []
            if sp:
                config = sp[0].get("config")
                split = sp[0].get("split", split)
        except Exception:
            pass

    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    path = os.path.join(OUT, name, "shard-000.jsonl")
    part = path + ".part"          # never clobber a good shard on failure
    written = 0
    rows = 0
    offset = 0
    h = hashlib.sha256()

    if dry_run:
        print(f"  [dry] {name}: would write up to {human(budget_left)}")
        return 0, 0, None

    with open(part, "wb") as f:
        while written < budget_left:
            url = (f"https://datasets-server.huggingface.co/rows?"
                   f"dataset={urllib.parse.quote(hf_id)}&split={split}"
                   f"&offset={offset}&length=100")
            if config:
                url += f"&config={urllib.parse.quote(config)}"
            payload = None
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(url, timeout=30) as r:
                        payload = json.load(r)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429 or e.code >= 500:
                        wait = 2 ** attempt
                        print(f"  [retry] {name}: HTTP {e.code}, sleeping {wait}s",
                              file=sys.stderr)
                        time.sleep(wait)
                        continue
                    print(f"  [warn] {name}: HTTP {e.code}", file=sys.stderr)
                    break
                except Exception as e:
                    print(f"  [warn] {name}: {type(e).__name__}: {e}", file=sys.stderr)
                    break
            if payload is None:
                break
            time.sleep(0.25)
            batch = payload.get("rows") or []
            if not batch:
                break
            for row in batch:
                row_data = row.get("row", {})
                text = row_data.get(field)
                if not isinstance(text, str) or len(text) < 64:
                    continue
                line = json.dumps({"text": text}, ensure_ascii=False).encode() + b"\n"
                f.write(line)
                h.update(line)
                written += len(line)
                rows += 1
                if written >= budget_left:
                    break
            offset += 100

    if rows == 0:
        os.unlink(part)
        return 0, 0, None
    os.replace(part, path)

    man = {
        "id": hf_id, "config": config, "split": split, "field": field,
        "licence": licence, "category": category, "why": why,
        "bytes": written, "rows": rows, "sha256": h.hexdigest(),
        "source": "datasets-server.huggingface.co/rows",
    }
    with open(os.path.join(OUT, name, "MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=1)
    return written, rows, man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-mb", type=int, default=400)
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list:
        for c in CATALOG:
            print(f"{c[0]:16s} {c[1]:40s} {c[5]:18s} {c[7]}")
        print("\nrefused:")
        for k, v in REFUSED.items():
            print(f"  {k}: {v}")
        return 0

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    budget = args.budget_mb * 1024 * 1024
    os.makedirs(OUT, exist_ok=True)
    total = 0
    index = []

    for name, hf_id, config, split, field, licence, category, why in CATALOG:
        if only and name not in only:
            continue
        left = budget - total
        if left <= 0:
            print(f"[stop] budget {human(budget)} exhausted")
            break
        print(f"[fetch] {name} ({licence}, up to {human(left)})")
        if name in RAW:
            url, delim = RAW[name]
            written, rows, man = fetch_raw(name, url, delim, licence, category, why,
                                           left, args.dry_run)
            if rows:
                total += written; index.append(man)
                print(f"        {rows} rows, {human(written)}")
            continue
        written, rows, man = fetch_one(name, hf_id, config, split, field,
                                       licence, category, why, left, args.dry_run)
        if rows:
            total += written
            index.append(man)
            print(f"        {rows} rows, {human(written)}")

    if index and not args.dry_run:
        with open(os.path.join(OUT, "INDEX.json"), "w") as f:
            json.dump({"budget_bytes": budget, "total_bytes": total,
                       "datasets": index, "refused": REFUSED}, f, indent=1)
    print(f"[done] {human(total)} in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
