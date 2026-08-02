# COORDINATION LEDGER — concurrent-agent mutual exclusion

This file prevents THIS agent (cog) and the OTHER concurrently-working agent from
both mutating the same source file. Protocol (from CoAgent MTPO / MCP playbook):

1. Before editing any `src/` or `tools/` or `Makefile` file, append a LOCK line.
2. After finishing, append an UNLOCK line.
3. If a file is LOCKED by the other agent, pick a different stone or wait.
4. Heartbeat every step; stale locks (>30 min) may be adopted.

## Current locks
(append below; format: LOCK <agent> <file> <iso-ts> <eta-min>)

## History
- 2026-08-01T__:__ cog seeded AH theme + COORDINATION.md (research/INDEX.md, COORDINATION.md, research/AGI_HOME_METAGAME.md) — UNLOCKED after write.

## cog (this agent) scope
- Research synthesis, new C11 primitives, operator dims, ledger.
- Avoids editing wubu_model.c core internals and recursive_optimize.c core loop
  unless explicitly unlocked by the other agent.
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615356 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615356
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615357 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615357
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615361 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615361
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615361 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615361
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615361 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615361
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615364 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615364
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615364 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615364
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615368 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615368
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615368 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615368
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615368 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615368
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615368 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615368
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615508
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615856 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615856
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615872
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615878 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615878
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615878 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615878
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615878 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615878
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615878 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615878
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615878 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615878
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615918
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
LOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971 10
UNLOCK recursive_optimize operator_applied.json,optimizestate.json,dgm_archive.json 1785615971
