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
| KERNEL | 92 | 26 | 33 | 1 | 32 |
| BRAIN | 112 | 43 | 30 | 9 | 30 |
| UX | 95 | 53 | 21 | 0 | 21 |
| AGENT | 70 | 30 | 19 | 0 | 21 |
| DRIVERS | 50 | 0 | 20 | 9 | 21 |
| NAMESPACE | 65 | 22 | 11 | 25 | 7 |
| STACK | 45 | 18 | 10 | 8 | 9 |
| SECURITY | 52 | 0 | 10 | 27 | 15 |
| FS | 50 | 0 | 8 | 32 | 10 |
| PATHS | 32 | 18 | 7 | 0 | 7 |
| RUNTIME | 65 | 30 | 7 | 16 | 12 |
| WORLDSIM | 45 | 15 | 3 | 20 | 7 |
| NETWORK | 52 | 1 | 1 | 48 | 2 |
| OMARCHY | 35 | 21 | 0 | 13 | 1 |
| AGIEDR | 40 | 19 | 0 | 21 | 0 |
| **Total** | **1000** | **314** | **218** | **237** | **231** |

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

