#!/usr/bin/env python3
"""wubu_download_micro.py — fetch the curated micro-dataset pack.

Reads docs/micro-datasets.json, pulls every file of each dataset (skipping
README/.gitattributes), verifies the container magic, and writes a manifest.
Resumable: existing non-empty files are skipped.

  python3 tools/wubu_download_micro.py                 # all
  python3 tools/wubu_download_micro.py --only 5        # first 5
  python3 tools/wubu_download_micro.py --id NousResearch/hermes-function-calling-v1
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "docs", "micro-datasets.json")
API = "https://huggingface.co/api/datasets"
RESOLVE = "https://huggingface.co/datasets"
SKIP = {".gitattributes", "README.md", ".gitignore"}


def tree(ds):
    q = urllib.parse.quote(ds, safe="/")
    with urllib.request.urlopen(f"{API}/{q}/tree/main?recursive=true", timeout=60) as r:
        return [x for x in json.load(r) if x.get("type") == "file"]


def magic_ok(path):
    try:
        if os.path.getsize(path) < 64:
            return True  # tiny sidecar
        with open(path, "rb") as f:
            head = f.read(4)
        if path.endswith(".parquet"):
            return head == b"PAR1"
        if path.endswith(".json"):
            # pretty-printed JSON starts with '[' or '{' possibly followed by
            # a newline, so a line-based parse is wrong here.
            with open(path, "rb") as f:
                return f.read(64).lstrip()[:1] in (b"{", b"[")
        if path.endswith(".jsonl"):
            with open(path, "rb") as f:
                first = f.readline().strip()
            if not first:
                return False
            json.loads(first)
            return True
        return True
    except Exception:
        return False


def fetch(url, dest, retries=3):
    for a in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wubuwizard/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest + ".part", "wb") as f:
                while True:
                    c = r.read(1 << 20)
                    if not c:
                        break
                    f.write(c)
            os.replace(dest + ".part", dest)
            return True
        except Exception as e:
            for p in (dest + ".part",):
                if os.path.exists(p):
                    os.remove(p)
            if a == retries:
                print(f"    FAIL {os.path.basename(dest)}: {e}")
                return False
            time.sleep(2 * a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", action="append", default=[])
    ap.add_argument("--only", type=int, default=0)
    ap.add_argument("--max-mb", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    spec = json.load(open(SPEC))
    out_root = spec["target_dir"]
    todo = spec["datasets"]
    if args.id:
        todo = [d for d in todo if d["id"] in set(args.id)]
    if args.only:
        todo = todo[:args.only]

    os.makedirs(out_root, exist_ok=True)
    index = []
    for d in todo:
        ds, slug = d["id"], d["id"].replace("/", "__")
        dest_dir = os.path.join(out_root, slug)
        os.makedirs(dest_dir, exist_ok=True)
        files = [f for f in tree(ds) if os.path.basename(f["path"]) not in SKIP]
        total = sum(f.get("size", 0) for f in files)
        if total > args.max_mb * 1024 * 1024:
            print(f"[skip] {ds}: {total/1e6:.0f} MB > cap {args.max_mb} MB")
            continue
        print(f"[{ds}] {len(files)} files, {total/1e6:.1f} MB")

        jobs = []
        for f in files:
            dest = os.path.join(dest_dir, f["path"])
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            url = f"{RESOLVE}/{urllib.parse.quote(ds, safe='/')}/resolve/main/{urllib.parse.quote(f['path'])}?download=true"
            jobs.append((url, dest))

        ok = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch, u, p): p for u, p in jobs}
            for fut in as_completed(futs):
                p = futs[fut]
                if fut.result():
                    if magic_ok(p):
                        ok += 1
                    else:
                        os.remove(p)
                        print(f"    BAD  {os.path.basename(p)} (magic) — removed")
        have = sum(os.path.getsize(os.path.join(dest_dir, f["path"]))
                   for f in files if os.path.exists(os.path.join(dest_dir, f["path"])))
        entry = {"id": ds, "category": d["cat"], "license": d["license"],
                 "fields": d.get("fields", []), "files": len(files),
                 "bytes": have, "path": dest_dir,
                 "source": f"https://huggingface.co/datasets/{ds}"}
        with open(os.path.join(dest_dir, "MANIFEST.json"), "w") as f:
            json.dump(entry, f, indent=1)
        index.append(entry)
        print(f"  -> {have/1e6:.1f} MB present, {ok} fetched this run")

    # merge with the existing index so a partial run does not clobber it
    prev = {}
    ipath = os.path.join(out_root, "INDEX.json")
    if os.path.exists(ipath):
        try:
            for e in json.load(open(ipath)).get("datasets", []):
                prev[e["id"]] = e
        except Exception:
            pass
    for e in index:
        prev[e["id"]] = e
    allv = sorted(prev.values(), key=lambda x: x["id"].lower())
    with open(ipath, "w") as f:
        json.dump({"wave": spec["wave"], "datasets": allv,
                   "total_bytes": sum(e["bytes"] for e in allv),
                   "eval_only": spec.get("eval_only", [])}, f, indent=1)
    print(f"\npack: {len(allv)} datasets, {sum(e['bytes'] for e in allv)/1e6:.1f} MB -> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())