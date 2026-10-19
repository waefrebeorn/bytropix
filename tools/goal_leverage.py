#!/usr/bin/env python3
"""Rank open GOALS-1000 goals by module cluster (the leverage view).

Most goals name a module file. Several goals usually share one file, so building
that file once can close the whole cluster. This tool groups open goals by
target module, marks whether the file already exists, and prints the clusters
worth doing first.

Usage:
  tools/goal_leverage.py            # ranked table
  tools/goal_leverage.py --json     # machine-readable
  tools/goal_leverage.py --min 3    # only clusters with >= 3 open goals
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPOS = [ROOT, "/home/wubu/wubuos", "/home/wubu/wubunos"]
SKIP = {".git", "build", "node_modules", "target", ".cache", "research"}


def index_basenames():
    """Basenames of every .c/.h in the three repos (bounded walk)."""
    names = set()
    for root in REPOS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP]
            for f in filenames:
                if f.endswith((".c", ".h", ".S")):
                    names.add(f)
    return names


def is_done(g):
    return g["title"].strip().endswith("(DONE)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--min", type=int, default=2)
    args = ap.parse_args()

    path = os.path.join(ROOT, "docs", "GOALS-1000.json")
    data = json.load(open(path))
    goals = data["goals"] if isinstance(data, dict) else data
    present = index_basenames()

    open_goals = [g for g in goals if not is_done(g)]
    by_mod = collections.defaultdict(list)
    for g in open_goals:
        by_mod[g.get("module") or "?"].append(g)

    clusters = []
    for mod, gs in by_mod.items():
        base = os.path.basename(mod.strip())
        grounded = base in present
        clusters.append({
            "module": mod,
            "open": len(gs),
            "grounded": grounded,
            "domains": dict(collections.Counter(g["domain"] for g in gs)),
            "tiers": dict(collections.Counter(g["tier"] for g in gs)),
            "goals": [g["n"] for g in gs],
            "titles": [g["title"] for g in gs],
        })

    clusters.sort(key=lambda c: (-c["open"], c["grounded"], c["module"]))
    big = [c for c in clusters if c["open"] >= args.min]

    summary = {
        "total": len(goals),
        "open": len(open_goals),
        "done": len(goals) - len(open_goals),
        "target_modules": len(clusters),
        "grounded_open": sum(1 for g in open_goals
                             if os.path.basename((g.get("module") or "").strip()) in present),
        "clusters_at_min": len(big),
        "goals_in_clusters_at_min": sum(c["open"] for c in big),
        "singletons": sum(1 for c in clusters if c["open"] == 1),
    }

    if args.json:
        print(json.dumps({"summary": summary, "clusters": big}, indent=1))
        return 0

    print(f"goals {summary['total']}  open {summary['open']}  done {summary['done']}")
    print(f"grounded open {summary['grounded_open']}  "
          f"target modules {summary['target_modules']}  singletons {summary['singletons']}")
    print(f"\nclusters with >= {args.min} open goals: {len(big)} "
          f"covering {summary['goals_in_clusters_at_min']} goals\n")
    print(f"{'open':>4}  {'file':<8}  module")
    for c in big:
        doms = ",".join(f"{k}:{v}" for k, v in sorted(c["domains"].items(), key=lambda kv: -kv[1]))
        print(f"{c['open']:>4}  {'yes' if c['grounded'] else 'MISSING':<8}  {c['module']}  [{doms}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
