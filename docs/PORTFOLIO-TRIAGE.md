# Portfolio triage — cycle 2 (2026-10-18)

Every one of the 1,000 registry goals is classified. The full per-goal
machine-readable status is `python3 tools/goal_portfolio.py --json`; this page
is the cycle's published summary.

Rules used:

- **done** — title carries `(DONE)`, and the module + test evidence exists.
- **active** — open, module exists on disk, tier P0/P1. Candidate work now.
- **blocked** — open, but the module named by the goal does not exist in any of
  the three repos. Reason: not written yet. A blocked goal cannot be closed by
  prose.
- **deferred** — open, module exists, tier P2/P3. Deferred with expiry
  **2026-12-31**; if still deferred then, it must be promoted or dropped with a
  written reason.

| Domain | Total | Done | Active | Blocked | Deferred |
|---|---:|---:|---:|---:|---:|
| COMPILER | 100 | 18 | 38 | 8 | 36 |
| KERNEL | 92 | 27 | 32 | 1 | 32 |
| BRAIN | 112 | 43 | 30 | 9 | 30 |
| UX | 95 | 53 | 21 | 0 | 21 |
| AGENT | 70 | 30 | 19 | 0 | 21 |
| DRIVERS | 50 | 1 | 19 | 9 | 21 |
| NAMESPACE | 65 | 22 | 11 | 25 | 7 |
| STACK | 45 | 18 | 10 | 8 | 9 |
| SECURITY | 52 | 3 | 7 | 27 | 15 |
| FS | 50 | 0 | 8 | 32 | 10 |
| PATHS | 32 | 18 | 7 | 0 | 7 |
| RUNTIME | 65 | 30 | 7 | 16 | 12 |
| WORLDSIM | 45 | 17 | 1 | 20 | 7 |
| NETWORK | 52 | 1 | 1 | 48 | 2 |
| OMARCHY | 35 | 21 | 0 | 13 | 1 |
| AGIEDR | 40 | 19 | 0 | 21 | 0 |
| **Total** | **1000** | **321** | **211** | **237** | **231** |

## What the split says

- 237 blocked goals all name modules that do not exist. That is the honest size
  of the unbuilt backlog, and it is concentrated in NETWORK (48), FS (32),
  SECURITY (27), NAMESPACE (25) and WORLDSIM (20).
- Only 2 NETWORK goals are active: the 9P work lives mostly under NAMESPACE.
- Four domains have zero DONE (DRIVERS, SECURITY, FS, NETWORK). Two have zero
  active goals (OMARCHY, AGIEDR): their remaining work is blocked or deferred.
  Zero done in domains where modules already exist is the strongest signal
  that nobody has been working there.

## Cycle 2 disposition

- Active work: AGENT #336 (closed), BRAIN #244 (advanced).
- Deferred expiry: 2026-12-31 for all 231 P2/P3 goals.
- No goal was silently dropped; every open goal appears in exactly one of
  active / blocked / deferred.

## Cycle 3 disposition

- Domains: AGENT, STACK (BRAIN sat out — it had run two cycles in a row).
- AGENT #333 closed: DA-1 static contradiction screen plus a budget test
  proving zero evals are spent on an unsatisfiable declaration set.
- STACK #773 advanced, not closed: strict-C11 audit now reports 197 sources,
  386 diagnostics and **0 errors** (10 fixed). The empty-diagnostic criterion
  still fails, so the goal stays open.

## Cycle 4 disposition

- Domains: COMPILER, STACK (AGENT and STACK had run twice; AGENT sat out,
  STACK continued a second cycle).
- COMPILER #114 closed: lexer integer suffixes now classify 32/64-bit and
  unsigned/signed, including hex and binary forms; `test_lexer_suffix` 13/13
  in the wubunos `test` target.
- STACK #773 advanced again: diagnostics 386 → **162**, files 32 → **13**,
  errors remain 0. Implicit-declaration family eliminated; braced-groups and
  the `GQA_KV_DIM` redefinition remain.

## Cycle 5 disposition

- Domains: UX, NAMESPACE (both fresh; neither had run in the previous cycle).
- UX #495 closed: `test_dosgui_startmenu` 12/12 after fixing five real defects —
  prefix-before-substring ranking, an uninitialized `lower_query` terminator
  (single-char queries matched nothing), a self-aliasing `strncpy` (UB at -O3),
  Tab/Esc mode desynchronisation, and a launch stub that could not observe the
  action.
- NAMESPACE #403 closed: Tclunk now rejects unknown or already-clunked fids
  with Rerror and leaves the fid table untouched; `test_styx` 31/31.
- Body gate: `make test` run and recorded in the ledger.

## Cycle 6 disposition

- Domains: NETWORK, STACK (STACK resumed after one cycle out).
- NETWORK #865 closed: `styx_server_t` now carries a negotiation ceiling
  (`max_msize`, 0 = `STYX_MAX_MSG`); 8192 against a 4096 server negotiates 4096;
  a 5000-byte TWRITE gets `Rerror` and the stream stays in sync for the next
  message. `test_styx` 32/32.
- STACK #773 advanced: guarding the duplicate `GQA_KV_DIM` definition cut
  pedantic diagnostics **162 → 105** across **13 → 6** files; errors remain 0.
  Ten braced-groups and a few const/plugin warnings remain, so it stays open.

## Cycle 11 disposition

- Domains: WORLDSIM, SECURITY (BRAIN sat out after two consecutive cycles).
- WORLDSIM #567 **closed**: `ws_collision_t` + `ws_collision_allows()` +
  `ws_query_layer_pairs()` in `physics.c`. The test covers mutual pairing,
  mask-0 isolation, a filter change taking effect on the immediate next query,
  the 16th bit, max_pairs capping and NULL arguments. `test_worldsim` **44/44**.
- SECURITY #910 **closed**: `wubu_cap_audit.c` adds a 256-record denial ring
  (pid, kind, requested rights, failing check, monotonic seq). Wired into every
  refusal path in `wubu_cap_resolve` (audience, rights, revoked). Exactly 100
  denials produce exactly 100 records; 300 more wrap at capacity while the total
  keeps counting. `test_cap` **41/41**.
- **Real bug found by the work**: the audience arrays are scanned to a
  `WUBU_PID_NONE` terminator with no length argument, so any shorter array is a
  stack overread. ASan caught `wubu_cap_system_install_to_pid()` passing a
  one-element array — production code, not just the test. Fixed there and in the
  test callers, and the contract is now written in `wubu_cap.h`. ASan is clean
  on the whole suite.

## Cycle 10 disposition

- Domains: BRAIN, WORLDSIM.
- WORLDSIM #564 **closed**: `src/worldsim/rng.c` implements PCG-XSH-RR
  (64-bit LCG, XSH-RR output, integer-only). `test_worldsim` now runs the
  statistical battery: pinned 8-word vector, 1e6-sample chi-square over 256 bins
  (passes the 1% critical value 310.46), 32-bit run check with bit balance, and
  a **10k-value FNV hash `0xf4b6ad7df17336c3`** as the cross-arch guard.
  37/37 checks.
- BRAIN #266 **advanced, not closed**: the test was running at `D_MODEL=0`
  because `WUBU_DIMS` was never initialized — every loop was a no-op and every
  gradient check read an empty buffer. With `wubu_dims_default()` the suite is
  **15/15**. But the criterion asks for a **finite-difference check at 1e-4**
  and zero gradients on unrouted tokens; neither exists yet, and the current
  gradients reach **~1e9**, which is a saturation smell that an FD oracle would
  probably catch. Leaving it open with the dims fix landed is the honest move.

## Cycle 9 disposition

- Domains: BRAIN, STACK (both had been idle for several cycles).
- BRAIN 1006 closed: orbital mechanics over the existing gravity cells —
  Kepler period, 1:2 resonance, Poincaré distance, cooling migration to the core
  floor. `test_orbit` 22/22, circular orbit drifts 0.26% over 5,000 steps, wired
  into `test_all`.
- STACK 1007 closed: pollution ledger with the honest baseline — **28 datasets,
  0 measured, 6 eval_only**. A spectral line is refused without a recorded
  before/after eval; `--check` is now part of `test_all` as `ledger-check`.
- **Red test found while looking for a third closure:** BRAIN #266
  (`test_nested_moe_router_backward`) passes only **4/15**. Every gradient
  assertion fails (`d_x`, `d_coarse`, `d_fine` all zero), and the NULL-pointer
  cases inherit the same failure. This is recorded as a real defect, not closed
  and not relabelled. It is the strongest BRAIN candidate for cycle 10.
- **Duplication caught:** `wubu_gravity.h` (2026-08-04, INDEX AN12) already owns
  gravitating cells and potential routing. `wubu_orbit` was re-scoped to
  mechanics only, with the overlap documented in the header rather than left as
  a second router.

## Cycle 8 disposition

- Domains: KERNEL, SECURITY. KERNEL's #29 was chosen from the reality check as
  a genuine build, not a relabel; SECURITY was refreshed with a stronger test.
- KERNEL #29 closed with new code: `src/kernel/wubu_pt.c` implements a real
  4-level walker over a caller-supplied physical arena — PML4 → PDPT → PD → PT,
  4 KiB and 2 MiB leaves, PS-bit leaves, recursive-free unmap, and refusal of a
  4 KiB map under a large leaf or a 2 MiB map over an existing table. The test
  hand-walks every level and asserts P/RW/US bits: **34/34**.
- SECURITY #911 closed: the criterion claims four aliases / five handles, but
  the test only built one alias. The test now builds all four, revokes through
  the first, and asserts all five stop resolving: **348/348**.
- Correction to the reality check: KERNEL #29 was listed as nominal. It was
  nominal *because nobody had written it*; it is now written. The other KERNEL
  rows (buddy, bump handoff, callee-saved switch) remain nominal.

## Cycle 7 disposition

- Domains: DRIVERS, SECURITY — both fresh, bringing the touched set to **9 of
  16** (BRAIN, STACK, AGENT, COMPILER, UX, NAMESPACE, NETWORK, DRIVERS,
  SECURITY).
- DRIVERS #956 closed: `input_ps2_feed()` implements the set-2 wire format —
  0xF0 break, E0 extended, and the six-byte E1 1D 45 E1 9D C5 Pause sequence as
  exactly one event. `test_input` 12/12.
- SECURITY #908 closed **by verification, not new code**: `test_vsl_nt` already
  asserts the cap ceiling refuses a right the DACL allows and reports a zero
  grant (336/336). Recorded as verified so the registry stops implying it is
  unproven.

