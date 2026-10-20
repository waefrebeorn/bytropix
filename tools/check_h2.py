#!/usr/bin/env python3
"""check_h2.py -- refuse a Harness^2 run that adopts something it did not earn.

Rules enforced on every LEDGER.jsonl record:
  1. adopted >= 0 requires a probe with that index;
  2. that probe's check_rc must be 0;
  3. its verdict must be "helped" (a candidate that merely matched a passing
     baseline changes nothing, and several-edits-at-once is "confounded");
  4. it must carry at least one rule;
  5. the task must have a library EXPERIENCE.md next to the ledger.

Usage: python3 tools/check_h2.py --root runs/harness2 [--self-test]
Exit 0 when every record is honest, 1 otherwise (with one line per violation).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile


def audit(root: str) -> list[str]:
    bad: list[str] = []
    ledger = os.path.join(root, "LEDGER.jsonl")
    if not os.path.exists(ledger):
        return [f"no ledger at {ledger}"]
    with open(ledger, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                bad.append(f"line {ln}: not JSON ({exc})")
                continue
            adopted = rec.get("adopted", -1)
            probes = rec.get("probes", [])
            if adopted < 0:
                continue
            match = [p for p in probes if p.get("i") == adopted]
            if not match:
                bad.append(f"line {ln}: adopted probe {adopted} not in record")
                continue
            p = match[0]
            if p.get("check_rc") != 0:
                bad.append(f"line {ln}: adopted probe {adopted} did not pass "
                           f"(check_rc={p.get('check_rc')})")
            if p.get("verdict") != "helped":
                bad.append(f"line {ln}: adopted probe {adopted} verdict="
                           f"{p.get('verdict')!r}, expected 'helped'")
            if p.get("rules", 0) < 1:
                bad.append(f"line {ln}: adopted probe carries no rule")
            task = rec.get("task", "")
            exp = os.path.join(root, "library", task, "EXPERIENCE.md")
            if task and not os.path.exists(exp):
                bad.append(f"line {ln}: adopted task {task!r} has no {exp}")
    return bad


def self_test() -> int:
    fails = 0

    def check(cond, label):
        nonlocal fails
        print(("PASS  " if cond else "FAIL  ") + label)
        fails += 0 if cond else 1

    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "library", "t-ok"), exist_ok=True)
    open(os.path.join(root, "library", "t-ok", "EXPERIENCE.md"), "w").write("cue: x\n")
    good = {"task": "t-ok", "adopted": 1, "probes": [
        {"i": 0, "rules": 0, "check_rc": 1, "verdict": "no-effect"},
        {"i": 1, "rules": 1, "check_rc": 0, "verdict": "helped"}]}
    with open(os.path.join(root, "LEDGER.jsonl"), "w") as f:
        f.write(json.dumps(good) + "\n")
    check(audit(root) == [], "accepts an adopted 'helped' run with a lesson")

    bad_conf = {"task": "t-ok", "adopted": 1, "probes": [
        {"i": 0, "rules": 0, "check_rc": 1, "verdict": "no-effect"},
        {"i": 1, "rules": 3, "check_rc": 0, "verdict": "confounded"}]}
    with open(os.path.join(root, "LEDGER.jsonl"), "w") as f:
        f.write(json.dumps(bad_conf) + "\n")
    v = audit(root)
    check(any("confounded" in x for x in v), "rejects adoption of a confounded probe")

    bad_fail = {"task": "t-ok", "adopted": 1, "probes": [
        {"i": 0, "rules": 0, "check_rc": 1, "verdict": "no-effect"},
        {"i": 1, "rules": 1, "check_rc": 5, "verdict": "helped"}]}
    with open(os.path.join(root, "LEDGER.jsonl"), "w") as f:
        f.write(json.dumps(bad_fail) + "\n")
    check(any("did not pass" in x for x in audit(root)), "rejects adoption of a failing probe")

    bad_task = {"task": "t-missing", "adopted": 1, "probes": [
        {"i": 1, "rules": 1, "check_rc": 0, "verdict": "helped"}]}
    with open(os.path.join(root, "LEDGER.jsonl"), "w") as f:
        f.write(json.dumps(bad_task) + "\n")
    check(any("no " in x and "EXPERIENCE" in x for x in audit(root)),
          "rejects adoption without a lesson")

    print("ALL PASSED" if fails == 0 else f"{fails} FAILURES")
    return 0 if fails == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/harness2")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not os.path.isdir(a.root):
        print(f"no run directory at {a.root} (nothing to check)")
        return 0
    bad = audit(a.root)
    for b in bad:
        print("VIOLATION:", b)
    print("check_h2: %d violation(s)" % len(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())