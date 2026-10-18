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
| COMPILER | 100 | 17 | 39 | 8 | 36 |
| KERNEL | 92 | 26 | 33 | 1 | 32 |
| BRAIN | 112 | 43 | 30 | 9 | 30 |
| UX | 95 | 52 | 22 | 0 | 21 |
| AGENT | 70 | 29 | 20 | 0 | 21 |
| DRIVERS | 50 | 0 | 20 | 9 | 21 |
| NAMESPACE | 65 | 21 | 12 | 25 | 7 |
| STACK | 45 | 18 | 10 | 8 | 9 |
| SECURITY | 52 | 0 | 10 | 27 | 15 |
| FS | 50 | 0 | 8 | 32 | 10 |
| PATHS | 32 | 18 | 7 | 0 | 7 |
| RUNTIME | 65 | 30 | 7 | 16 | 12 |
| WORLDSIM | 45 | 15 | 3 | 20 | 7 |
| NETWORK | 52 | 0 | 2 | 48 | 2 |
| OMARCHY | 35 | 21 | 0 | 13 | 1 |
| AGIEDR | 40 | 19 | 0 | 21 | 0 |
| **Total** | **1000** | **309** | **223** | **237** | **231** |

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
