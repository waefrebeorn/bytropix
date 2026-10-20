#!/usr/bin/env python3
"""Harness² loop with Hermes as the executor.

Harness² (arXiv 2610.14224) refines the harness around a fixed model by
propose -> probe -> compose: an editor proposes harness edits, each candidate is
executed on the SAME task, and the harness is composed from the behavioural
contrast rather than from ground-truth labels. A second, persistent level records
lessons that guide the next task.

This driver binds that loop to the WuBu AGI loop:

  task        an id, a prompt, and a CHECK command that decides success
  harness     a directory with AGENTS.md (+ optional skills/) that is staged
  executor    `hermes -z <prompt> --in <workspace>` (HermesSubstrate)

A probe is honest only if the check can fail: the check command's exit status is
the score, and a candidate that does not improve it is discarded.

Usage
  python3 tools/harness2_agi.py --tasks tools/harness2_tasks.json --model <m> \
      --k 2 --out runs/harness2
  python3 tools/harness2_agi.py --tasks ... --dry-run     # no model calls
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
H2 = REPO / "third_party" / "harness2"
sys.path.insert(0, str(H2))

from harness2.substrate.hermes import HermesSubstrate  # noqa: E402

# Edits an agent-harness editor may apply. Each is a line appended to AGENTS.md;
# they are hypotheses, and only the check decides whether one helps.
EDITS = {
    "verify-first": "Before editing, run the project's check command and read its output.",
    "small-diff": "Change the fewest lines that satisfy the check; never reformat unrelated code.",
    "no-claims": "Do not report success unless the check command exited 0; quote its output.",
    "one-step": "Take one concrete step, then re-run the check before continuing.",
}

BASE_AGENTS = """# WuBu AGI harness

You are the executor inside the WuBu AGI loop. Work in the given workspace.

- Prefer measured facts over narration.
- A task is complete only when its check command exits 0.
"""


def build_harness(extra_rules: list[str]) -> Path:
    d = Path(tempfile.mkdtemp(prefix="h2-harness-"))
    body = BASE_AGENTS
    if extra_rules:
        body += "\n## Additional rules\n" + "\n".join(f"- {r}" for r in extra_rules) + "\n"
    (d / "AGENTS.md").write_text(body, encoding="utf-8")
    (d / "skills").mkdir(exist_ok=True)
    (d / "skills" / "README.md").write_text(
        "<!-- on-demand skills: skills/<name>/SKILL.md -->\n", encoding="utf-8"
    )
    return d


def run_check(cmd: str, cwd: Path, timeout: int) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        return 124, f"[check timed out after {timeout}s]"
    except OSError as exc:
        return 127, f"[check failed to start: {exc}]"


def probe(task: dict, rules: list[str], model: str, workroot: Path,
          timeout: int, dry: bool) -> dict:
    harness = build_harness(rules)
    ws = workroot / task["id"] / (("+".join(rules) if rules else "baseline").replace(" ", "_"))
    ws.mkdir(parents=True, exist_ok=True)
    # Stage the harness next to the workspace so the agent can read it.
    shutil.copytree(harness, ws / ".harness", dirs_exist_ok=True)
    sub = HermesSubstrate()
    prompt = task["prompt"] + f"\n\nHarness instructions are in {ws / '.harness' / 'AGENTS.md'}."
    record = {"task": task["id"], "rules": rules, "workspace": str(ws)}
    if dry:
        record["argv"] = sub.run_argv(prompt, ws, model)
        record["rc"] = None
        record["check_rc"] = None
        return record
    t0 = time.time()
    rc, out, err = sub.run_once(prompt, ws, model, timeout=timeout)
    record.update(rc=rc, seconds=round(time.time() - t0, 1),
                  stdout_tail=out[-1500:], stderr_tail=err[-800:])
    if rc != 0:
        record["check_rc"] = rc if rc else 1
        record["check_note"] = "executor failed before the check ran"
        return record
    check_rc, check_out = run_check(task["check"], Path(task.get("cwd", ws)), timeout)
    record.update(check_rc=check_rc, check_tail=check_out)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--k", type=int, default=2, help="candidate edits per task")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="runs/harness2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = json.load(open(args.tasks))["tasks"]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    edit_names = list(EDITS)[: max(0, args.k)]

    summary = []
    for task in tasks:
        tdir = outdir / task["id"]
        tdir.mkdir(parents=True, exist_ok=True)
        probes = [probe(task, [], args.model, tdir / "probes", args.timeout, args.dry_run)]
        for name in edit_names:
            probes.append(probe(task, [EDITS[name]], args.model, tdir / "probes",
                                args.timeout, args.dry_run))
        passed = [p for p in probes if p.get("check_rc") == 0]
        winner = passed[0] if passed else None
        adopted = winner["rules"] if winner else []
        summary.append({"task": task["id"], "probes": len(probes),
                        "passed": len(passed), "adopted": adopted})
        (tdir / "run.json").write_text(json.dumps(
            {"task": task, "probes": probes, "adopted": adopted}, indent=1), encoding="utf-8")
        print(f"{task['id']}: {len(passed)}/{len(probes)} probes passed"
              f"{' — adopted ' + ','.join(adopted) if adopted else ''}")

    (outdir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0 if all(s["passed"] for s in summary) or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())