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

## Runnable C11 driver

```
make harness2_c11
./harness2_c11 <task-id> <ledger.jsonl> <check-cmd> [rule ...]
```

Baseline probe plus one probe per rule, each scored by the check; adoption is the
first passing probe and one JSONL record is appended. Measured:

```
./harness2_c11 demo  /tmp/h2ledger.jsonl 'test -f /tmp/h2_marker' 'noop rule A'
  probe 0 check_rc=1 · probe 1 check_rc=1 · compose: no probe passed -> rc 1
./harness2_c11 demo2 /tmp/h2ledger.jsonl 'true' 'rule A' 'rule B'
  probe 0 check_rc=0 · adopted probe 0 · ledger now holds 2 records
```

The Python driver below is a reference implementation of the same loop; the
system path is the C11 one.

## Usage (reference)

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
## Devil's advocate: three rounds, and what each changed

### Round 1 — algorithm fidelity

**Criticism.** The paper's composition runs on *behavioural contrast without a
ground-truth verifier* (§2.2.1: "without ground-truth feedback", "contrasts inform
composition rather than verify task correctness"). The C11 loop scores probes by a
check command's exit status. That is a verifier — a different regime, and exactly
what the paper says does not exist in the open world. Calling the module "the
two-level loop" overstated it.

**Also under-modelled:** two candidates per step (invariant + domain-specific),
k-way parallel or sequential recursion, selective merge/revise composition, a
final agentic selector, and an improvement harness holding `EXPERIENCE.md` lessons
with edit attribution (`helped | hurt | no observed effect | confounded`).

**Implemented.** Attribution now exists and is honest about its regime:
`wubu_h2_verdict()` returns HELDED/HURT/NO_EFFECT/CONFOUNDED, and a probe carrying
more than one edit is reported **CONFOUNDED** rather than credited. The ledger
records `result` and `verdict` per probe.

**Still different, on purpose:** selection is deterministic (first passing probe,
baseline included) and there is no invariant/domain split, no k-way scheduling, no
agentic final selector. Those need a model in the loop; the controller does not
pretend to have one.

### Round 2 — measurement credibility

**Criticism.** The compiler numbers may have counted tests that never ran.

**Found.** Real defect: without `.PHONY`, `make test_X` on an already-built target
printed "up to date" and skipped `./$@`. Fixed in the body repo for all 36 targets;
`make test` then still reported 200/200 — so the claim moved from assumed to
verified. The gauntlet is unaffected (it forks and executes directly).

**Also measured:** raising `HD_MAX_PARAMS` 16 → 64 grew `HDASTNode` to **9,592
bytes** (inline `param_names[64][64]` + two pointer arrays). Peak RSS compiling the
largest suite file: **4.6 MB**. Accepted, with the cost recorded rather than
hidden.

### Round 3 — implementation defects

Four, all fixed:

| defect | fix |
|---|---|
| `wubu_h2_adopt` promised rule text, returned only a count, and never wrote `out` | copies the winning harness's rules (the probe now carries a `harness` pointer) |
| the C11 driver freed each harness before adoption read it — dangling pointer | harnesses are kept alive until after compose/adopt |
| `check_tail` was a dead field; no output was captured | pipe captures the **last** bytes the check printed |
| ledger refused any string containing a quote (lost lessons) | proper JSON escaping, including control characters |

And one robustness gap found while probing the timeout path: `alarm()` kills the
shell but not its children. The child now gets its own process group and the
parent `kill(-pid, SIGKILL)`s the group when the alarm fires.

**Known gap left open:** probes have no `cwd` or environment restriction — a check
runs in the caller's directory with the caller's environment. Recorded here rather
than papered over.

`make test_harness2`: **75 checks, ALL PASSED** after these changes.
