# Harness² in the WuBu AGI loop

Upstream: <https://github.com/google-research/harness2> — *Harness²: Recursive Agent
Harnessing for an Open World* (arXiv 2610.14224). Apache-2.0.

Vendored at `third_party/harness2/`, pinned by `third_party/harness2/UPSTREAM_SHA`
(`1cfe254e8d25b15f37048c3730134686abe60335`). Licence text travels with the copy.

## Why it matters here

Harness² improves the *system around* a fixed model — prompts, tools, memory,
skills, orchestration — by contrast rather than by labels:

- **task-level recursion**: propose harness edits → probe them on the same task →
  compose a harness from the behavioural contrast (no ground-truth feedback);
- **harnessing-level recursion**: persist adopted harnesses and evidence-supported
  lessons so later tasks start from what worked.

That is the same shape as our standing loop (measure, adopt what moved a number,
discard what did not), so it is used as the controller rather than reinvented.

## Execution path: C11, not Python

The loop that runs is **`src/wubu_harness2.c`** (`include/wubu_harness2.h`):

| function | what it owns |
|---|---|
| `wubu_h2_harness_add_rule/render/stage` | the editable rule set and the exact bytes of `AGENTS.md` (deterministic) |
| `wubu_h2_probe_run` | runs the task's **check command** with `alarm()` timeout; 124 on timeout, never success |
| `wubu_h2_compose` | first passing probe wins, baseline included |
| `wubu_h2_adopt` | the adopted rule count for the winning harness |
| `wubu_h2_ledger_append/count` | append-only JSONL lessons, refusing strings that would not parse |
| `wubu_h2_executor_argv` | `hermes -z <prompt> --in <ws> [-m model] --pass-session-id` |

`make test_harness2` runs **58 checks** (rules, capacity, determinism, staging,
probe exit codes and timeout, composition order, adoption, ledger append/parse,
argv shape and overflow). The vendored Python tree under `third_party/harness2/`
is **reference only** — it documents the upstream design and is not executed by
the loop.

## What was added

1. `harness2/substrate/hermes.py` — **HermesSubstrate** (`name = "hm"`):
   - `surface()` — Hermes-native harness tree: `AGENTS.md` always-on,
     `skills/<name>/SKILL.md`, `scripts/*.py`;
   - `materialise()` — stages the tree plus `file:` context entries;
   - `run_argv()` — `hermes -z <prompt> --in <workspace> [-m model] --pass-session-id`;
   - `run_once()` — one non-interactive turn, returns `(rc, stdout, stderr)`;
   - `read_trajectory()` — clipped transcript read-back;
   - `preflight()` — fails loudly if the `hermes` binary is missing.
   Tests: `python3 third_party/harness2/tests/test_hermes_substrate.py` (12 checks).
2. `tools/harness2_agi.py` — the loop driver: for each task, run a baseline probe
   and up to `--k` edit probes, score each by a **check command that can fail**,
   adopt the rules of the first probe that passes, write `run.json` per task and
   `summary.json` for the run.
3. `tools/harness2_tasks.json` — WuBu tasks with real checks (`make
   test_mir_battery` → WRONG 0; `make test_timer_wheel` → 0 failures).

The upstream benchmark adapters (`--bench jb|wb|lab`, substrates `oc|cx`) are left
untouched: they carry benchmark-specific launchers. WuBu drives the substrate
directly.

## Usage

```bash
# no model calls: prints the argv each probe would run
python3 tools/harness2_agi.py --tasks tools/harness2_tasks.json --model <m> --k 2 --dry-run

# real probes (spends model tokens and wall time)
python3 tools/harness2_agi.py --tasks tools/harness2_tasks.json --model <m> --k 2 \
    --timeout 900 --out runs/harness2
```

Task schema:

```json
{"id": "compiler-battery",
 "prompt": "...what the agent must do...",
 "check": "shell command; exit 0 = success",
 "cwd": "/home/wubu/wubunos"}
```

## Measured end-to-end

`--k 0` (baseline probe only), default model, 2026-10-19:

```
compiler-battery: 1/1 probes passed        (executor rc=0, check rc=0, 65 s)
```

The first run returned 0/1 because the check itself was wrong — `make
test_mir_battery` was printing `up to date` and never executing the binary. That
was fixed in the body repo (`.PHONY` for all 36 test targets); the check then
passed. A probe is only evidence once its check can fail.

## Invariants

- **A probe that cannot fail is not a probe.** Every check exits non-zero on the
  broken state; the driver records `check_rc`, not the model's self-report.
- **Adoption is by measured contrast**, never by the executor claiming success.
- **Model weights stay fixed.** Only the harness changes.
- **Tokens and time are real**: each probe is a full Hermes turn; `--dry-run`
  exists so plumbing can be checked without spending either.

## Next

- Feed goal-registry entries (the 1000-gap backlog) as tasks, with `check`
  commands taken from each goal's criterion.
- Record adopted harnesses into the improvement harness (upstream `core/library.py`)
  once a run has more than one task with a passing probe.
- Wire `harness2_agi.py --k 1` into a nightly job so harness edits accumulate
  evidence instead of being re-proposed each session.