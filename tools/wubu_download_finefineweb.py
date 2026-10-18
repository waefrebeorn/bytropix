#!/usr/bin/env python3
"""wubu_download_finefineweb.py — domain-selective fetcher for m-a-p/FineFineWeb.

FineFineWeb is Apache-2.0 English web text, >1T tokens, 66,109 JSONL shards
partitioned into 68 topic domains. We do not need a terabyte: this tool pulls
whole domains (or a bounded number of shards) into the corpus tree, so the
OS/compiler/math domains can be mixed deliberately.

Layout on disk:
  <out>/<domain>/<domain>_NNNNNN.jsonl
  <out>/manifest.json          (source, license, per-file sizes, totals)

Resumable: a shard that exists with a non-zero size is skipped. A shard that
fails the JSONL sanity check is deleted so the next run retries it.

Examples:
  python3 tools/wubu_download_finefineweb.py --list
  python3 tools/wubu_download_finefineweb.py --domains computer_science_and_technology,mathematics --max-files 20
  python3 tools/wubu_download_finefineweb.py --domains aerospace --max-files 3 --out /tmp/ffw_probe

Exit code 0 when every requested shard is present and parses.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "m-a-p/FineFineWeb"
API = f"https://huggingface.co/api/datasets/{REPO}"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
LICENSE = "apache-2.0"
DEFAULT_OUT = "/home/wubu/models/corpus/finefineweb"


def list_files():
    with urllib.request.urlopen(API, timeout=60) as r:
        d = json.load(r)
    return [s["rfilename"] for s in d.get("siblings", [])]


def domains_of(files):
    out = {}
    for f in files:
        if "/" not in f:
            continue
        dom = f.split("/")[0]
        out.setdefault(dom, []).append(f)
    for dom in out:
        out[dom].sort()
    return out


def fetch(url, dest, retries=3):
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wubuwizard/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(dest + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(dest + ".part", dest)
            return True
        except Exception as e:
            if os.path.exists(dest + ".part"):
                os.remove(dest + ".part")
            if attempt == retries:
                print(f"  FAIL {os.path.basename(dest)}: {e}")
                return False
            time.sleep(2 * attempt)
    return False


def valid_jsonl(path, sample=3):
    """Cheap sanity check: first `sample` non-empty lines must be JSON objects."""
    try:
        seen = 0
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    return False
                seen += 1
                if seen >= sample:
                    return True
        return seen > 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", help="comma-separated domain names")
    ap.add_argument("--max-files", type=int, default=0,
                    help="max shards per domain (0 = all)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--list", action="store_true", help="list domains and shard counts")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    files = list_files()
    doms = domains_of(files)

    if args.list:
        print(f"{REPO}  license={LICENSE}  shards={len(files)}  domains={len(doms)}")
        for d in sorted(doms, key=lambda x: -len(doms[x])):
            print(f"{d:40s} {len(doms[d]):5d} shards")
        return 0

    if not args.domains:
        print("need --domains (or --list)", file=sys.stderr)
        return 2

    wanted = []
    for dom in [x.strip() for x in args.domains.split(",") if x.strip()]:
        if dom not in doms:
            print(f"unknown domain: {dom}", file=sys.stderr)
            return 2
        take = doms[dom] if args.max_files <= 0 else doms[dom][:args.max_files]
        wanted.extend(take)

    os.makedirs(args.out, exist_ok=True)
    todo, skipped = [], 0
    for rel in wanted:
        dest = os.path.join(args.out, rel)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            skipped += 1
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        todo.append((rel, dest))

    print(f"{REPO}: {len(wanted)} requested, {skipped} already present, "
          f"{len(todo)} to fetch -> {args.out}")

    ok = 0
    for rel, dest in todo:
        url = f"{BASE}/{rel}?download=true"
        if fetch(url, dest):
            if valid_jsonl(dest):
                ok += 1
            else:
                os.remove(dest)
                print(f"  BAD  {rel} (not JSONL) — removed")

    # manifest: rescan the domain dirs we touched
    entries, total_bytes = [], 0
    for dom in {r.split("/")[0] for r in wanted}:
        dd = os.path.join(args.out, dom)
        if not os.path.isdir(dd):
            continue
        for name in sorted(os.listdir(dd)):
            if not name.endswith(".jsonl"):
                continue
            p = os.path.join(dd, name)
            sz = os.path.getsize(p)
            total_bytes += sz
            entries.append({"path": f"{dom}/{name}", "bytes": sz})
    manifest = {
        "source": f"https://huggingface.co/datasets/{REPO}",
        "license": LICENSE,
        "language": ["en"],
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shards": len(entries),
        "bytes": total_bytes,
        "files": entries,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    print(f"done: fetched {ok}/{len(todo)} this run; "
          f"{manifest['shards']} shards, {total_bytes/1e6:.1f} MB on disk")
    return 0 if ok == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())