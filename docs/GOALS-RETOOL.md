# GOALS-1000 retool — backlog regenerated from evidence

Date: 2026-10-19. Operator instruction: *"retool the 1000 goals to be fresh gaps"*,
choosing regeneration over rewriting the open subset. DONE history is therefore
discarded; the registry is a **backlog**, not a ledger.

## Invariant kept

Exactly **1000 goals**, exactly **16 domains**, `n` contiguous 1..1000, every
criterion at least 25 characters and phrased as something a command can check.
The previous registry is frozen at `docs/GOALS-1000.baseline.json` so the
generator never consumes its own output.

## What each entry is

| source | count | what the gap is | how it is checked |
|---|---|---|---|
| `Build …` | 235 | the module named by a previously-open goal **does not exist** in any of the four repos | file appears; `make test_<stem>` exists and prints ALL PASSED |
| `Verify …` | 403 | the module exists, so the **capability is unverified**, not absent | evidence recorded in `docs/PORTFOLIO-REALITY-CHECK.md` with file:line, owning test green |
| `Test coverage: …` | 265 | module ships with **no test target** at all | a `test_<stem>` target exists, exercises the entry point, prints ALL PASSED |
| `Parse class: …` | 70 | a measured gauntlet failure class (70 classes, 550 captured tests) | every test in the class builds; the class count in `docs/GAUNTLET-CLASSES.json` reaches 0 |
| `Runtime answers: …` | 5 | tests that compile but exit with the wrong code (210 such tests) | they report PASS, each fix naming wrong and expected values |
| feature gaps | 22 | capabilities measured absent this session | named assertion per entry |

Per-domain counts are a consequence of the evidence, not a quota:
BRAIN 151, COMPILER 146, KERNEL 124, UX 121, RUNTIME 85, NETWORK 51, SECURITY 49,
DRIVERS 49, FS 48, AGENT 41, NAMESPACE 30, STACK 28, WORLDSIM 28, AGIEDR 21,
OMARCHY 14, PATHS 14.

## Regenerate

```
python3 tools/retool_registry.py --dry-run   # print counts, write nothing
python3 tools/retool_registry.py             # rewrite docs/GOALS-1000.json
```

Inputs: `docs/GOALS-1000.baseline.json`, `docs/GAUNTLET-CLASSES.json`,
`docs/GAUNTLET-FAILURES.json`, the four repo trees (module resolution and
test-target inventory). The run is idempotent: re-running produces a byte-identical
file because the baseline is frozen.

## What the numbers say

The backlog is now dominated by work that was previously invisible:

- **235 modules are simply missing** — the old registry called them active.
- **403 capabilities are unverified** — a module with a matching filename proved
  nothing, which is what `PORTFOLIO-REALITY-CHECK.md` was created to record.
- **265 modules have no regression test**, so "green gate" never covered them.

Only 75 of 1000 entries are gauntlet-derived, and 22 are the feature gaps measured
this session (`_Complex`, ELF emission, spans, computed goto, f32 storage,
`#if`, variadic macros, distortion fit, GrabCut, Farneback, volume pipeline,
pretrain corpus, A2A bind, QEMU scanout, WM draw proof, stack watermark, 9P
socket, VFS writeback, TCP RTO, Start-menu keys, strict-C11 burn-down, printf).

## Honesty notes

- "Verify" entries assert that evidence is *recorded*, not that the capability is
  broken: some may already work and simply lack a proof. Recording the proof is
  the work.
- Coverage entries name modules that ship without tests; that is a gap in
  assurance, not evidence of a defect.
- Nothing here is auto-closable. Closing still requires a test target printing
  ALL PASSED and the criterion evidence written down.
