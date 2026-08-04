# research/056 — THE HIVE DIAGNOSTIC SYSTEM (the superior AGI diagnostic)

> 2026-08-04. The user's directive: "we need to design the superior AGI
> diagnostic system using our hive structure. The hive will save us."
> Doctrine: THE HIVE IS THE BODY (wubu_amoeba) — cells ARE hive slots,
> GROW = hive_insert (mitosis), SHRINK = hive_erase (apoptosis),
> DIAGNOSE = hive_foreach (jumps skips, sees only live).

## The one-sentence design

**The hive is not just where the colony lives — the hive IS the diagnostic
system.** Every measurement the AGI takes (loss, grads, route, oracle
score, system vitals) is a CELL in the same hive tissue as the model's own
cells. Diagnosis = walking that tissue; mutation = growing/shrinking that
tissue; the 5+1 recovery = replaying that tissue. One structure, every
level, memory-bounded by construction — the hive cannot bloat.

## Why this is SUPERIOR to the current immune system

The current `wubu_amoeba_diagnose()` is a **one-shot classifier**: it
measures the CURRENT step's grads and decides grow/shrink/stasis. It has
no memory, no trends, no causality. The superior system makes the hive a
**diagnostic trace**:

| | Current (one-shot) | Superior (the hive trace) |
|---|---|---|
| Memory | none (a snapshot) | the hive's fixed blocks — a bounded ring of recent measurements |
| Trend | single measurement | z-score over the live window (mean/std per kind) |
| Cause | "the colony is unhealthy" | the causal walker finds the FIRST anomalous cell before a fitness drop |
| Scope | cell grads only | LOSS / GRAD / ENTROPY / ROUTE / UTIL / BI / ORACLE / DATA / SYS / MUT — the whole stack |
| Bloat | (the checkpoint problem we just fixed!) | ring-bounded by the hive's capacity — erase = skip + freelist push, O(1) |
| Rollback | 5+1 blind restore | the walker tells the 5+1 WHAT to roll back |

## The cell kinds (the hive's measurement vocabulary)

Each hive slot = one typed measurement: `{ kind, step, cell, value, meta }`.

| kind | measures | source |
|---|---|---|
| `LOSS` | train / held-out loss + ema | the trainer (every step) |
| `GRAD` | per-cell grad norms | the REAL backprop (wubu_backprop — the milestone) |
| `ENTROPY` | route entropy per layer | the router (moe2/hashrouter) |
| `ROUTE` | which cells fired per token | the forward — THE route trace = the model's self-explanation |
| `UTIL` | per-cell utilization | the router's counter |
| `BI` | block importance | wubu_bi |
| `ORACLE` | RLHF / NVIDIA score_draft | the live loop (nvidia_nim) |
| `DATA` | corpus stream health (which tier fed what) | the corpus mixer |
| `SYS` | disk / RAM / GPU / SSD free (the foundry's vitals) | the tensor store + PowerShell probe |
| `MUT` | the mutation ledger: grow/shrink + fitness delta | the amoeba operators |

## The ring discipline (memory-bounded by construction)

- Insert a measurement = `wubu_hive_insert` (freelist pop or new block).
- Erase stale = `wubu_hive_erase` (skip-mark + LIFO push) — when the hive
  is full, the OLDEST cells recycle automatically.
- Foreach = `wubu_hive_foreach` (jumps skips, sees only live).
- Capacity = hive blocks × slots (e.g. 64 slots/block × N blocks). The
  trace keeps the last ~K epochs of measurements — NEVER the whole
  history. **The diagnostic memory cannot bloat** (the lesson of the
  103-checkpoint / 15 GiB archive is baked into the structure).

## The anomaly detector (the immune system, over the window)

Per kind, the hive maintains aggregate state over the live window:
- **mean / std / min / max** — updated incrementally on insert.
- **z-score** per measurement: `(x − mean)/std`. `|z| > 2.5` = anomalous.
- **grad health**: the quadratic approximation `ΔL ≈ −‖g‖²/n` (the loss
  delta a cell's gradient predicts) + the ABSOLUTE floor (1e-4 — the DA
  bug: relative-only misses the all-dead colony).
- **route entropy** (uniform → healthy; degenerate → stuck).
- Classification per cell, per window:
  - **grow candidate** — grad z-score rising across the window AND
    utilization above mean (overworked: the pseudopod extends).
  - **shrink candidate** — below the absolute floor for the WHOLE window
    (dead: the membrane retracts).
  - **stasis** — everything else.

## The causal walker (root-cause diagnosis — the superior move)

On a fitness drop (held-out loss rises past `loss_tol`):
1. Record the drop as a `LOSS` cell (the event).
2. Walk the hive BACKWARD from the drop, skipping erased cells
   (`hive_foreach` is already skip-aware).
3. Find the EARLIEST out-of-family measurement (z > threshold) that
   precedes the drop — the root cause candidate.
4. Emit the diagnosis: `"at step N kind=GRAD cell=7 z=+4.2 preceded the
   loss rise at N+50; cell 7 is overworked → grow"` — or the honest
   `"no out-of-family measurement found; fitness drop unexplained"`.

The 5+1 recovery consumes the walker's report: roll back to the last
accepted state BEFORE the first anomalous measurement, not a blind
restore. The walker is what makes mistakes SAFE and SPECIFIC.

## The DGM wiring (the loop, one tissue)

```
TRAIN      -> LOSS/GRAD/ROUTE/ENTROPY/UTIL cells inserted every step
DIAGNOSE   -> the detector over the window + the walker on demand
MUTATE     -> grow/shrink = hive_insert/hive_erase (the body morphs)
             -> a MUT cell records {op, cell, fitness_before, fitness_after}
VALIDATE   -> held-out loss (LOSS cell) + the Lean prover (geometry intact)
             + the tensor store (checkpoint materializable in any format)
COMMIT     -> accepted: the MUT lineage ledger stays in the hive;
             rejected: the walker + 5+1 rollback (wubu_recovery)
REPEAT     -> the hive is a different size; the trace remembers why
```

## The double meaning (self-knowledge, preserved)

1. The model diagnoses ITSELF — the detector over GRAD/UTIL/ENTROPY cells.
2. The model diagnoses THE WORLD — the ROUTE cells are the reasoning
   trace: which cells fired, in what order, for what input. That trace IS
   the transparent explanation.
The two are the SAME foreach. **The route trace is the diagnosis; the
diagnosis is the model's self-knowledge.**

## The OS integration (the Body hosts the Brain's diagnosis)

- `wubuos/src/kernel/wubu_agi_kernel.c` already has a trace ring — the
  diagnostic hive is its memory substrate (the metal port of wubu_hive).
- Expose the hive at the Styx/9P namespace `/n/diag/`:
  - `status` — the colony's vitals (aggregate window stats)
  - `cells` — the live measurement trace (foreach dump)
  - `walker` — the latest root-cause report
  - `mutate` — write the grow/shrink command (the control plane)
- The tensor store (`wubu_tensor_store`, research/055 AN07) gives the
  walker + 5+1 instant access to ANY checkpoint in ANY format (the
  rollback materializes via the uniform catalog — no conversion waste).

## The module sketch (the implementation wave)

```
include/wubu_diag.h
  typedef enum { WUBU_DIAG_LOSS, WUBU_DIAG_GRAD, WUBU_DIAG_ENTROPY,
                 WUBU_DIAG_ROUTE, WUBU_DIAG_UTIL, WUBU_DIAG_BI,
                 WUBU_DIAG_ORACLE, WUBU_DIAG_DATA, WUBU_DIAG_SYS,
                 WUBU_DIAG_MUT } wubu_diag_kind;
  typedef struct { wubu_diag_kind kind; int64_t step; int cell;
                   float value; float meta; } wubu_diag_cell;

  wubu_diag_t *wubu_diag_init(wubu_hive_t *hive, int kinds);
  int wubu_diag_record(wubu_diag_t*, wubu_diag_kind, int cell,
                       float value, float meta);   /* hive_insert */
  int wubu_diag_zscore(const wubu_diag_t*, wubu_diag_kind, float value);
  int wubu_diag_classify(wubu_diag_t*, float *grow, float *shrink);
  int wubu_diag_walk(wubu_diag_t*, int64_t drop_step, char *report,
                     size_t cap);                  /* the causal walker */
  int wubu_diag_snapshot(wubu_diag_t*, const char *json_path);
src/wubu_diag.c  — window aggregates (mean/std), z-scores, the walker,
                   the per-kind floors, MUT ledger append
tools/test_diag.c — DA oracles:
  1. ring-bounded: inserting > capacity recycles the oldest (foreach
     sees exactly capacity live cells; the freelist recycles)
  2. z-score: an injected 10x outlier has |z| > 2.5; a normal value < 1
  3. trend: a cell whose grad rises across 20 steps classifies GROW
  4. dead colony: all grads below the floor classifies SHRINK (the DA bug)
  5. the walker: build a trace with a normal window, inject an anomaly at
     step N, inject a fitness drop at N+50; the walker reports kind/cell/N
  6. honest failure: a drop with no anomaly reports "unexplained"
```

## The DA review (the design's own triple devil's-advocate)

- **DA-1 correctness**: z-score + incremental mean/std is standard;
  the walker's earliest-anomaly rule is a documented heuristic with an
  honest "unexplained" fallback. The per-kind absolute floors are
  configurable. The classification thresholds are the SAME in diagnose
  and mutate (the DA bug from the amoeba — one source of truth).
- **DA-2 privacy**: everything is local; the ORACLE kind is opt-in (the
  NVIDIA key is the user's; no telemetry).
- **DA-3 robustness**: the ring bounds memory by construction; a full
  hive degrades to "keep the newest K" (the freelist recycles); the
  walker never crashes on an empty/unexplained trace — it reports.

## Milestones (in order)

1. `wubu_diag` module + tests (the sketch above) — the hive as trace.
2. Wire the REAL per-cell grads from wubu_backprop into GRAD cells
   (the amoeba's milestone 1, now on the trace).
3. The walker → 5+1 wiring (wubuos wubu_recovery consumes the report).
4. `/n/diag/` Styx export (the Body hosts the Brain's diagnosis).
5. The MUT lineage ledger → the DGM branch tree (every variant, its
   fitness, its parent — the hive's own archive).

## Registration

- INDEX theme AN, entry AN08 (this doc): `open` → `wired` when
  wubu_diag + test_diag land.
- The hive is in BOTH repos (reference + metal port) — the boundary
  contract; the diagnostic hive inherits it.
