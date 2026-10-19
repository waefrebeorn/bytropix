# GOAL LAYOUT REASSESSMENT — maximise closures per unit of real work

2026-10-18 · generated from `docs/GOALS-1000.json` by `tools/goal_leverage.py`

## Measured state

- **1,000 goals · 321 done · 679 open** (16 domains)
- **511 distinct target modules** for 679 open goals — the registry is file-anchored,
  which is good: a goal names the file that will satisfy it.
- **420 open goals point at a file that exists**; 259 point at a file that does not.
- **32 clusters hold ≥3 open goals each, 149 goals total.** The other 428 open
  goals are singletons (one goal, one module).

Domain spread of the open work: COMPILER 82, BRAIN 69, KERNEL 65, NETWORK 51,
FS 50, SECURITY 49, DRIVERS 49, NAMESPACE 43, UX 42, AGENT 40, RUNTIME 35,
WORLDSIM 28, STACK 27, AGIEDR 21, OMARCHY 14, PATHS 14.

## Diagnosis

The old rotation rule ("advance ≥1 goal in ≥2 domains per cycle, touch ≥8 domains
per 8 cycles") maximised *breadth*. It actively worked against closure:

1. It forces domain switching every cycle, so a 9-goal cluster in `tasking.c` gets
   one goal per visit and the file is re-opened nine times.
2. It treats a singleton and a 9-goal cluster as equal-cost, which they are not.
3. It never retires anything, so 9 duplicate title groups and 259 ungrounded goals
   keep re-entering the plan.

The registry's own arithmetic says the same thing: 1000 goals over 511 modules
means **the unit of work is the module, not the goal**.

## Leverage table (clusters ≥3, from `tools/goal_leverage.py --min 3`)

Highest-value first. "exists" means the target file is already in the tree; a
MISSING file is a build, but its whole cluster lands at once.

| open | file | exists | domains |
|-----:|------|:------:|---------|
| 9 | `wubu_isa_x86_64.c` | yes | COMPILER 9 |
| 9 | `wubuos/src/kernel/tasking.c` | yes | KERNEL 8, SECURITY 1 |
| 8 | `holyd_parse.c` | yes | COMPILER 8 |
| 8 | `styx_serve.c` | yes | NAMESPACE 4, FS 3, NETWORK 1 |
| 8 | `styxfs_callbacks.c` | yes | NAMESPACE 6, KERNEL 2 |
| 7 | `holyd_mir_eval.c` | yes | COMPILER 7 |
| 6 | `vsl/vsl_nt_ob.c` | **no** | NAMESPACE 6 |
| 5 | `wubunos/Makefile` | **no** | STACK 5 |
| 5 | `wubu_ns_root.c` | **no** | NAMESPACE 5 |
| 5 | `wubu_mir_opt.c` | yes | COMPILER 5 |
| 5 | `wubu_mir_regalloc.c` | yes | COMPILER 5 |
| 5 | `styxfs_server.c` | yes | KERNEL 3, SECURITY 2 |
| 4 | `wubu_ns_audit.c` / `wubu_ns_mount.c` / `wubu_socket.c` | **no** | NAMESPACE 6, NETWORK 3 |
| 4 | `ahci.c`, `wubu_drv_nvme.c`, `wubu_pci.c`, `wubu_iommu.c` | yes | DRIVERS 12, KERNEL 4 |
| 4 | `memory.c`, `wubu_crash.c`, `wubu_console_colonel.c` | yes | KERNEL 12 |

## Reallocation: six batches

Each batch is one *module* or one tight module family, with its own test binary.
A batch closes its whole cluster or it does not ship.

**B1 — NAMESPACE foundation (5 new files, ~20 goals).**
`vsl/vsl_nt_ob.c` (6), `wubu_ns_root.c` (5), `wubu_ns_audit.c` (3),
`wubu_ns_mount.c` (3), `kernel/wubu_socket.c` (3). These files do not exist, so
every goal in them is blocked on the same thing: a namespace core with mount
table, audit ring and socket shim. One coherent build unlocks 20 goals and is the
largest single win available.

**B2 — KERNEL scheduling + memory (existing files, ~21 goals).**
`tasking.c` (9), `memory.c` (4), `wubu_crash.c` (4), `wubu_console_colonel.c` (4).
Scheduler semantics are testable in the hosted build; the console and crash paths
have harnesses already.

**B3 — COMPILER front end (existing files, ~24 goals).**
`wubu_isa_x86_64.c` (9), `holyd_parse.c` (8), `holyd_mir_eval.c` (7). This is the
largest cluster in the registry and the compiler repo already has a working MIR
interpreter to test against.

**B4 — STYX / 9P surface (existing files, ~21 goals).**
`styx_serve.c` (8), `styxfs_callbacks.c` (8), `styxfs_server.c` (5). One server
test can cover several of these; the 9P namespace skill already describes the
pattern.

**B5 — DRIVERS (existing files, ~16 goals).**
`ahci.c` (4), `wubu_drv_nvme.c` (4), `wubu_pci.c` (4), `wubu_iommu.c` (4). Unit
tests with synthetic register maps; no hardware required for probe logic.

**B6 — Verification sweep of grounded singletons (unknown, likely 10–30).**
420 grounded open goals include capabilities that already exist but were never
marked: the wave-22 CV work is exactly this pattern (a module that had the
function, a goal that said it did not). A read-only audit of the top 50 grounded
singletons, checking the named capability against the actual function table,
should close a batch of them by *verification*, not by new code — with the same
rule as before: only when the criterion is met, never because a nearby test passes.

## Policy changes

1. **Cluster-first rotation replaces domain rotation.** Work one cluster to
   closure; the domain tally is a *consequence*, not a target. The 8-of-16 rule is
   retired.
2. **The unit of progress is a module with a test binary.** A cluster of N goals
   needs one `test_*` target and N criterion checks inside it.
3. **MISSING-file goals are scheduled as builds, not goals.** They are blocked
   until the file exists, and they all unblock together — B1 exists precisely
   because five missing files hold 20 goals.
4. **Retire duplicates.** Nine duplicate title groups should be merged into their
   canonical numbering rather than tracked twice.
5. **Verification sweep is a legitimate batch.** Closing a goal because the
   capability demonstrably exists is cheaper than rebuilding it, but it requires
   the criterion to be checked, not inferred.

## Expected shape

Clusters ≥3 hold 149 goals in 32 modules. Working strictly cluster-first, the
next 32 module-builds cover 149 goals (22% of the open set) with 32 test targets
instead of 149 increments. The remaining 428 singletons stay as the long tail and
are best handled by the same verification sweep before any new code is written.

## Reproduce

```
python3 tools/goal_leverage.py            # ranked clusters
python3 tools/goal_leverage.py --json     # machine-readable
python3 tools/goal_leverage.py --min 3    # what the table above used
```

## Update 2026-10-18 — pillar precedence (supersedes the batch order below)

The six batches below were ordered by cluster size. They are now ordered by the
design instead: **GEO → BODY → KVFS → ROUTE → REV → MANIFOLD**. Membership is
unchanged; only precedence and two tiers moved (292 Möbius backward → P0,
931 KV-FS session keys → P1). See `docs/GOAL-PILLARS.md`.

Also fixed at the same time: the model is **not** "35M". It is a
runtime-dimensioned body wearing either the checkpoint-native geometry
(35,073,216) or the aligned geometry (56,376,832) — `docs/MODEL-IDENTITY.md`.
