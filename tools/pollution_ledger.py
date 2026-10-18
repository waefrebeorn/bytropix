#!/usr/bin/env python3
"""Pollution ledger: which corpus dyed which line of WuBu STAR.

The star's colour is accretion. A model's identity is its corpus. This tool
keeps the mapping honest: every dataset the pack knows about is listed, and a
spectral line is claimed only when someone records a real before/after eval.

Commands
  --scan                 rebuild the dataset list from docs/micro-datasets.json
                         (preserves recorded evals and lines)
  --record ID --eval NAME --before X --after Y [--line NM]
                         append a measured delta; refuses non-finite values
  --check                validate the ledger (exit 1 on any violation)

Nothing here invents numbers. A dataset with no eval stays `unmeasured` and its
line stays null, which is the truthful state for the whole pack today.
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "micro-datasets.json")
LEDGER = os.path.join(ROOT, "docs", "POLLUTION-LEDGER.json")


def load_json(path, default=None) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else (default or {})
    except FileNotFoundError:
        return default or {}


def save(led):
    with open(LEDGER, "w") as f:
        json.dump(led, f, indent=1, sort_keys=False)
        f.write("\n")


def scan(led):
    pack = load_json(SRC, {})
    eval_only = set(pack.get("eval_only", []))
    prior = {d["id"]: d for d in led.get("datasets", [])}

    out = []
    for ds in pack.get("datasets", []):
        did = ds.get("id")
        if not did:
            continue
        old = prior.pop(did, {})
        out.append({
            "id": did,
            "category": ds.get("cat", "unknown"),
            "licence": ds.get("license", "unknown"),
            "eval_only": did in eval_only,
            "line_nm": old.get("line_nm"),
            "status": old.get("status", "unmeasured"),
            "evals": old.get("evals", []),
        })
    # Keep entries that are not in the pack (e.g. FineFineWeb, Jev) so recorded
    # work is never silently dropped.
    out.extend(prior.values())
    led["datasets"] = out
    led["counts"] = {
        "datasets": len(out),
        "measured": sum(1 for d in out if d.get("status") == "measured"),
        "eval_only": sum(1 for d in out if d.get("eval_only")),
    }
    return led


def record(led, did, name, before, after, line):
    for v in (before, after):
        if not math.isfinite(v):
            sys.exit("refusing non-finite eval value")
    if before < 0 or after < 0:
        sys.exit("refusing negative eval value")
    if not name.strip():
        sys.exit("refusing an unnamed eval")
    if line is not None and not (300.0 <= line <= 900.0):
        sys.exit("refusing a line outside 300-900 nm")

    ds = next((d for d in led.get("datasets", []) if d["id"] == did), None)
    if ds is None:
        sys.exit(f"unknown dataset: {did} (run --scan first)")

    ds.setdefault("evals", []).append({
        "name": name,
        "before": before,
        "after": after,
        "delta": after - before,
    })
    ds["status"] = "measured"
    if line is not None:
        ds["line_nm"] = line
    led["counts"]["measured"] = sum(1 for d in led["datasets"]
                                    if d.get("status") == "measured")
    return led


def check(led):
    errors = []
    pack = load_json(SRC, {})
    known = {d["id"] for d in pack.get("datasets", [])}
    eval_only = set(pack.get("eval_only", []))

    seen = set()
    for ds in led.get("datasets", []):
        did = ds.get("id")
        if not did:
            errors.append("entry without id")
            continue
        if did in seen:
            errors.append(f"duplicate dataset {did}")
        seen.add(did)
        if did not in known:
            errors.append(f"{did}: not in micro-datasets.json (orphan)")
        if ds.get("eval_only") and ds.get("trained"):
            errors.append(f"{did}: eval_only dataset marked trained")
        for ev in ds.get("evals", []):
            if not ev.get("name"):
                errors.append(f"{did}: eval without a name")
            b, a, d = ev.get("before"), ev.get("after"), ev.get("delta")
            if not all(isinstance(x, (int, float)) and math.isfinite(x)
                       for x in (b, a, d)):
                errors.append(f"{did}/{ev.get('name')}: non-finite eval")
                continue
            if abs((a - b) - d) > 1e-9:
                errors.append(f"{did}/{ev.get('name')}: delta != after-before")
        if ds.get("status") == "measured" and not ds.get("evals"):
            errors.append(f"{did}: measured without any eval")
        if ds.get("status") == "unmeasured" and ds.get("line_nm") is not None:
            errors.append(f"{did}: line claimed while unmeasured")
        if did in eval_only and ds.get("line_nm") is not None and ds.get("status") != "measured":
            errors.append(f"{did}: eval_only dataset claims a line without measurement")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dataset")
    ap.add_argument("--eval")
    ap.add_argument("--before", type=float)
    ap.add_argument("--after", type=float)
    ap.add_argument("--line", type=float)
    a = ap.parse_args()

    led = load_json(LEDGER, {
        "star": "WuBu STAR",
        "reference": "WD J1330+6435",
        "note": ("A line is claimed only with a recorded before/after eval. "
                 "Unmeasured datasets are provenance, not colour."),
        "datasets": [],
        "counts": {},
    })

    if a.scan or not (a.record or a.check):
        led = scan(led)
        save(led)
        print(f"scanned {led['counts']['datasets']} datasets "
              f"({led['counts']['measured']} measured, "
              f"{led['counts']['eval_only']} eval_only)")

    if a.record:
        if not (a.dataset and a.eval and a.before is not None and a.after is not None):
            sys.exit("--record needs --dataset --eval --before --after")
        led = scan(led)
        led = record(led, a.dataset, a.eval, a.before, a.after, a.line)
        save(led)
        print(f"recorded {a.dataset}: {a.eval} {a.before} -> {a.after}")

    if a.check:
        errs = check(led)
        if errs:
            print("LEDGER INVALID:")
            for e in errs:
                print("  -", e)
            return 1
        print(f"POLLUTION LEDGER OK: {led['counts'].get('datasets', 0)} datasets, "
              f"{led['counts'].get('measured', 0)} measured, "
              f"{led['counts'].get('eval_only', 0)} eval_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())