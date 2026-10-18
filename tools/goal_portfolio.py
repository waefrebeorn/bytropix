#!/usr/bin/env python3
"""goal_portfolio.py — portfolio controller for GOALS-1000.

Anti-myopia is a scheduling property, not a sentiment. This tool makes the
portfolio visible and proposes a round-robin work plan so one domain cannot
absorb every cycle.

Status convention: a goal is DONE when its title carries "(DONE)" (the
registry's existing convention); everything else is OPEN. A goal with no
module on disk is reported as UNGROUNDED — it cannot be closed by prose.

Usage:
  python3 tools/goal_portfolio.py            # human table + plan
  python3 tools/goal_portfolio.py --json     # machine-readable
  python3 tools/goal_portfolio.py --plan 8   # plan size (default 8)

Exit code 0 always (this is a reporting tool; the gate is the enforcer).
"""
import argparse
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(ROOT, "docs", "GOALS-1000.json")
TIER_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


SEARCH_DIRS = ["", "src", "include", "apps", "tools", "kernel", "drivers",
               "fs", "net", "ui", "lib"]
OTHER_ROOTS = ["/home/wubu/wubuos", "/home/wubu/wubunos",
               "/home/wubu/wubumath"]


def grounded(mod):
    """True if the module naming a goal resolves to a real file.

    Registry entries use every convention: 'wubu_kvfs.c', 'src/wubu_kvfs.c',
    'wubuos/src/kernel/foo.c', or several files joined by ' + '. We take the
    first token and try it relative to the brain, then the body/compiler/math
    roots, then by basename in each repo's usual source dirs."""
    first = re.split(r"[ +;,]", (mod or "").strip())[0].strip()
    if not first:
        return False
    roots = [ROOT] + OTHER_ROOTS
    # Registry paths often carry the repo name: "wubuos/src/kernel/x.c".
    # Strip the matching prefix before joining it to that repo's root.
    variants = [first]
    for name in ("wubuos/", "wubunos/", "wubuwizard/"):
        if first.startswith(name):
            variants.append(first[len(name):])
    for root in roots:
        for v in variants:
            if os.path.exists(os.path.join(root, v)):
                return True
    base = os.path.basename(first)
    if not base:
        return False
    for root in roots:
        for d in SEARCH_DIRS:
            if os.path.exists(os.path.join(root, d, base)):
                return True
    return False


def load():
    with open(REG) as f:
        d = json.load(f)
    goals = d["goals"] if isinstance(d, dict) else d
    for g in goals:
        g["done"] = "(DONE)" in g.get("title", "")
        g["grounded"] = grounded(g.get("module", ""))
    return goals


def summarize(goals):
    by_dom = defaultdict(lambda: {"total": 0, "done": 0, "p0_open": 0,
                                  "ungrounded": 0})
    for g in goals:
        s = by_dom[g["domain"]]
        s["total"] += 1
        if g["done"]:
            s["done"] += 1
        else:
            if g.get("tier") == "P0":
                s["p0_open"] += 1
            if not g["grounded"]:
                s["ungrounded"] += 1
    return dict(by_dom)


def plan(goals, n):
    """Round-robin across domains, P0 before P1, then registry order.
    One pick per domain per pass, so a single domain cannot dominate."""
    open_goals = [g for g in goals if not g["done"]]
    open_goals.sort(key=lambda g: (TIER_ORDER.get(g.get("tier"), 9), g["n"]))
    buckets = defaultdict(list)
    for g in open_goals:
        buckets[g["domain"]].append(g)
    # domains ordered by open P0 count desc, then name, for determinism
    doms = sorted(buckets, key=lambda d: (-sum(1 for g in buckets[d]
                                               if g.get("tier") == "P0"), d))
    out, i = [], 0
    while len(out) < n and any(buckets[d] for d in doms):
        d = doms[i % len(doms)]
        if buckets[d]:
            out.append(buckets[d].pop(0))
        i += 1
    return out


def check_cycle(path, window=8, min_domains=2, max_consecutive=2):
    """Validate the anti-myopia ledger. Each line is one cycle:
    {"cycle": N, "domains": [...], "gate": {"binaries":B,"passed":P,"fail":F},
     "closed": [ids], "deferred": {"DOMAIN": "reason (until YYYY-MM-DD)"}}
    Returns (ok, [problems])."""
    problems = []
    cycles = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    cycles.append(json.loads(line))
    except FileNotFoundError:
        return False, [f"no cycle ledger at {path}"]
    except json.JSONDecodeError as e:
        return False, [f"bad JSON in ledger: {e}"]

    for c in cycles:
        n = c.get("cycle", "?")
        doms = c.get("domains", [])
        if len(set(doms)) < min_domains:
            problems.append(f"cycle {n}: {len(set(doms))} distinct domain(s), "
                            f"need >={min_domains}")
        g = c.get("gate", {})
        if not g.get("binaries"):
            problems.append(f"cycle {n}: gate has no binary count")
        if g.get("fail", 1) != 0:
            problems.append(f"cycle {n}: gate failures = {g.get('fail')}")
        if not c.get("closed"):
            problems.append(f"cycle {n}: no goal closed (advance, don't drift)")

    # no domain may appear in more than max_consecutive cycles in a row
    streak = defaultdict(int)
    for c in cycles:
        doms = set(c.get("domains", []))
        for d in list(streak):
            if d not in doms:
                streak[d] = 0
        for d in doms:
            streak[d] += 1
            if streak[d] == max_consecutive + 1:
                problems.append(
                    f"cycle {c.get('cycle')}: domain {d} appears "
                    f"{streak[d]} cycles in a row (max {max_consecutive})")

    last = cycles[-window:]
    touched = set()
    for c in last:
        touched.update(c.get("domains", []))
    need = min(8, len(last))   # a young ledger cannot satisfy an 8-cycle rule
    if len(touched) < need:
        problems.append(f"last {len(last)} cycles touched {len(touched)} "
                        f"domains, need >={need} (anti-myopia floor)")

    return (not problems), problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--plan", type=int, default=8)
    ap.add_argument("--check-cycle", metavar="LEDGER",
                    help="validate a cycle ledger (anti-myopia rules)")
    args = ap.parse_args()

    if args.check_cycle:
        ok, problems = check_cycle(args.check_cycle)
        if ok:
            print(f"CYCLE LEDGER OK: {args.check_cycle}")
        else:
            print(f"CYCLE LEDGER FAILED: {args.check_cycle}")
            for p in problems:
                print("  -", p)
        raise SystemExit(0 if ok else 1)

    goals = load()
    by_dom = summarize(goals)
    total = len(goals)
    done = sum(1 for g in goals if g["done"])
    ungrounded = sum(1 for g in goals if not g["done"] and not g["grounded"])
    picks = plan(goals, args.plan)

    if args.json:
        print(json.dumps({
            "total": total, "done": done, "open": total - done,
            "ungrounded_open": ungrounded,
            "domains": by_dom,
            "plan": [{"n": g["n"], "domain": g["domain"], "tier": g["tier"],
                      "title": g["title"], "module": g["module"],
                      "grounded": g["grounded"]} for g in picks],
        }, indent=1))
        return

    print(f"PORTFOLIO: {total} goals | {done} DONE | {total - done} OPEN | "
          f"{ungrounded} ungrounded")
    print(f"{'domain':12s} {'total':>5s} {'done':>5s} {'open':>5s} "
          f"{'P0open':>7s} {'ungrounded':>10s}")
    for d in sorted(by_dom, key=lambda x: (-by_dom[x]["p0_open"], x)):
        s = by_dom[d]
        print(f"{d:12s} {s['total']:5d} {s['done']:5d} "
              f"{s['total'] - s['done']:5d} {s['p0_open']:7d} "
              f"{s['ungrounded']:10d}")
    print(f"\nANTI-MYOPIA PLAN ({len(picks)} picks, one per domain per pass):")
    for g in picks:
        flag = "" if g["grounded"] else "  [ungrounded]"
        print(f"  #{g['n']:<4d} {g['domain']:10s} {g['tier']}  "
              f"{g['title'][:78]}{flag}")


if __name__ == "__main__":
    main()