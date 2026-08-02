# Triple-DA Meta-Plan: the recursive gap-closing loop's own plan

Date: 2026-08-02. Auditor: the loop itself, applying the DA-3 discipline to the
META-PLAN (not to individual gaps). Context: the KB-7hop sweep just added 1004
gaps (themes IO-JC); passes 51-52 closed 48 of them; the user now asks for
triple-100 MORE research (AGI needs, Bonzi needs, cross-resource links) PLUS a
meta-plan. The DA asks: is the LOOP'S PLAN sound before we add more?

## DA-1 (correctness): the plan must close gaps faster than it creates them

- **Finding (REAL): the sweep created a backlog treadmill.** The sweep added
  1004 gaps in one turn; passes 51-52 closed 48 (4.8%). At one theme-batch per
  continue, the 956 remaining gaps take ~15 more batches, and each new
  research wave adds ~100 more. Without a CLOSE-RATE plan, the loop is a
  treadmill: open-gap count grows monotonically.
- **Verdict: the meta-plan MUST include a rate rule.** Rule M1: every research
  wave (a sweep OR a triple-100) must be accompanied by a CLOSE commitment of
  >= the wave's size, scheduled as the NEXT N passes. Rule M2: no new theme is
  appended while the open backlog exceeds 3x the close rate (a "backlog cap").
  This is the loop's own preemption guard (cf. IR04 activation-budget).
- **Finding (REAL): the sweep's gaps are not all REAL needs.** The sweep
  enumerated mechanisms from the LITERATURE (the KV survey, the Hopfield
  papers, the companion literature...) without checking each against the
  ENGINE's actual deficits. A gap is REAL only if the engine demonstrably
  lacks it AND a user-facing need drives it. The DA-1 fix: gaps must carry a
  "driver" field (which module deficit / which user need); gaps without a
  driver are deferred, not fabricated.

## DA-2 (privacy / safety / no-third-party): the plan's dependencies

- **Finding (REAL): the sweeps drift toward external dependencies.** Several
  swept mechanisms (PIM hardware, neuromorphic silicon, seccomp kernels,
  trained visual vocabs, RLHF datasets) require things the engine cannot
  self-host. The DA-2 rule: every gap must be closable as pure C11 with the
  existing 230-module codebase + zero external services (the standing rule).
  Gaps that are NOT (AX02, AX03, CC08, the PIM/SNN silicon items) stay marked
  "(research)" and never count toward the backlog.
- **Finding (REAL): the meta-plan lacks a resource map.** The user's ask —
  "from our new needs and existing resources to find more Kevin-Bacon links" —
  is the DA-2-consistent framing: the highest-leverage gaps are the ones that
  TIE MULTIPLE existing modules together (an integration gap), not the ones
  that need new infrastructure. The plan must PRIORITIZE BY CROSS-LINK COUNT.

## DA-3 (robustness): the plan must survive its own failure modes

- **Finding (REAL): the loop has no degradation path.** If a batch fails
  (build break, test red, collision with the sibling agent), the current plan
  has no fallback — it just stops. The meta-plan needs: per-batch rollback
  (git revert the batch), a collision protocol (check COORDINATION.md +
  `git status` before EVERY new module — the lesson from the wubu_ttc.h
  overwrite), and a "smallest-green-batch" rule (never commit a theme with a
  red suite).
- **Finding (REAL): the loop's own hyperparams are unmeasured.** The close
  rate, the per-batch gap count, the DA-caught-bug rate — none are tracked.
  The meta-plan adds a LOOP-LEDGER (rate, backlog, batches, DA-catches) that
  the loop consults before each pass — the recursive part.

## The meta-plan (adopted)

1. **Prioritize by cross-link count**: build the Theme JF resource map (new
   needs x existing modules) and close the highest-link gaps first.
2. **Backlog cap (M2)**: after this triple-100 wave, the open backlog must not
   exceed ~1,150; any further sweep must be paired with a close commitment.
3. **Driver field**: every NEW gap carries `(driver: <module-need | user-need>)`;
   driverless gaps are deferred.
4. **Research-marked gaps never count toward the close rate** (AX02/AX03/CC08
   + the silicon items in IS/IW stay honest-open).
5. **Loop-ledger**: track close-rate, DA-catches, and collisions; consult it
   before each batch.
6. **Bonzi is a first-class needs axis**: the human-facing GUI is the AGI's
   face (memory: "Bonzi + Comfy via WuBuFX, real dispatch not print theater");
   the Bonzi theme's driver is the USER-VISIBLE AGI, not the literature.

## What the triple-100 must satisfy (from this audit)

- JD (AGI meta-needs): every gap must map to an EXISTING engine module's
  deficit (driver = module), closing under the loop-ledger.
- JE (Bonzi needs): every gap must be real-dispatch (wubufx-renderable,
  WuBuOS-plumbable, C11-closable), NOT print theater.
- JF (cross-resource links): the Kevin-Bacon hop between the new needs and the
  230+ existing modules; these are the highest-leverage closes by definition.
