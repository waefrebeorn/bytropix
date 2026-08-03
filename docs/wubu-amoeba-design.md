# WUBU AMOEBA — the AGI-style diagnostic model (the model that evolves)

> 2026-08-03. The user: "design the AGI style diagnostic model using all
> of the research... a model that can evolve and get bigger and smaller
> like an amoeba. It is time."

## The one-sentence idea

WuBu is not a fixed-size network — it is a **colony**: a pool of
specialized cells (experts, neurons, blocks) watched by a **diagnostic
immune system** that measures each cell's contribution, then **grows**
(pseudopod: split the overworked cell), **shrinks** (apoptosis: prune
the dead cell, recycle its memory), or **stays** (stasis) — every
mutation empirically validated and archived, mistakes rolled back by
the 5+1 recovery. The model is an amoeba: it extends toward what it
needs, retracts what it doesn't, and its memory (the hive) recycles
every freed slot.

## Why this is the AGI way (the research synthesis)

| Research pillar | What it contributes | The amoeba mapping |
|---|---|---|
| **DGM / Darwin Gödel Machine** (AGI_HOME_METAGAME H3) | mutate → validate → archive; "bench, don't prove" | the grow/shrink loop: every mutation is a variant, kept only if fitness ↑ |
| **DeepSeekMoE fine-grained experts** (2401.02954) | many small experts, few active — expert specialization | the colony: cells are experts; the router is the immune system's eye |
| **The hive** (the user's diagram, wubu_hive) | linked blocks + skipfield + freelist — O(1) erase/reuse, stable pointers | the cell tissue: dead cells are skip-marked, slots recycled by the freelist — the amoeba's membrane |
| **The mixed agents** (wubu_moe2) | top-k routing with distinct agents | which cells fire per token |
| **The prover** (wubu_prover2, Lean-verified) | the sound reward signal — accepts/rejects proof steps | the fitness oracle: only verified growth survives |
| **The 5+1 recovery** (wubu_recovery) | five rollback slots + the Jesus state | mistakes are safe: a bad mutation rolls back to the last healthy colony |
| **Muon + the real backward pass** (wubu_backprop, RC01) | the correct training recipe | the growth engine: cells learn only when their gradients are real |
| **Gated-DeltaNet / linear attention** (008) | fixed-size state, linear KV | the colony's circulation: context flows through a small state, cheap to grow |
| **WuBu Nesting** (THEORY/03) | hyperbolic inter-level transitions | the colony's space: curvature changes = the amoeba's environment bends |
| **GRPO / RLVR** (043) | verifiable rewards, group-relative advantage | the colony's natural selection: fitness is measured, not assumed |
| **Data curriculum** (042) | stage-wise mixing + math annealing | the food supply: the colony grows fastest on the right diet |

## The architecture

```
┌────────────────────────── THE AMOEBA ──────────────────────────┐
│                                                                │
│   ┌──────────────┐    ┌──────────────────────────────────┐     │
│   │ DIAGNOSTIC   │    │ THE COLONY (the model body)      │     │
│   │ IMMUNE SYSTEM│───►│                                  │     │
│   │ (measures)   │    │  experts (wubu_moe2 cells)       │     │
│   └──────┬───────┘    │  ├─ cell 0  util 87%  ████████   │     │
│          │            │  ├─ cell 1  util  2%  ▏          │     │
│   per-cell metrics:   │  ├─ cell 2  util 45%  ████▌      │     │
│   - utilization       │  └─ cell N  ...                  │     │
│   - grad norm         │                                  │     │
│   - loss delta        │  memory: THE HIVE (skipfield +   │     │
│   - route entropy     │           freelist — the tissue) │     │
│          │            │  circulation: Gated-DeltaNet     │     │
│          ▼            │  space: WuBu Nesting (hyperbolic)│     │
│   ┌──────────────┐    └──────────────────────────────────┘     │
│   │ GROW / SHRINK│              ▲                              │
│   │ operators    │              │ mutations                    │
│   └──────┬───────┘              │                              │
│          ▼                      │                              │
│   ┌──────────────┐    ┌─────────┴──────────┐                  │
│   │ THE VERIFIER │    │ 5+1 RECOVERY slots │                  │
│   │ (prover +    │    │ (rollback if the   │                  │
│   │  held-out    │    │  mutation hurt)    │                  │
│   │  loss)       │    └────────────────────┘                  │
│   └──────┬───────┘                                            │
│          ▼                                                    │
│   THE ARCHIVE (the DGM branch tree: every accepted variant)   │
└────────────────────────────────────────────────────────────────┘
```

## The three operators (the amoeba's moves)

### 1. GROW — the pseudopod (mitosis)
**Trigger:** a cell's utilization or gradient norm exceeds the growth
threshold (it is overworked — the router keeps calling it).
**Action:** split the cell into two specialized daughters:
- `W_a = W + ε`, `W_b = W − ε` (the perturbation split)
- the router's gate for the parent is split into two gates
- new memory: two hive slots (the freelist provides them)
**Cost:** one checkpoint slot (rollback-safe).
**Effect:** the colony extends toward the overloaded region — the
amoeba's pseudopod reaches for the food.

### 2. SHRINK — apoptosis (retraction)
**Trigger:** a cell's utilization or gradient norm is below the death
threshold (it is dead weight — the router never calls it).
**Action:** prune the cell:
- its weights are archived (the variant is kept in the archive)
- its gate is removed from the router
- its hive slots are skip-marked + pushed to the freelist
  (the amoeba recycles the membrane — memory returns to the pool)
**Cost:** one checkpoint slot.
**Effect:** the colony retracts from the useless region — smaller,
faster, cheaper, no waste.

### 3. STASIS
**Trigger:** all cells within the healthy band.
**Action:** nothing. The colony keeps training (Muon + real backprop)
until the diagnosis says otherwise.

## The diagnostic metrics (what the immune system measures)

1. **Utilization** — per expert: routes received / total routes.
   High = overworked (grow), low = dead (shrink).
2. **Gradient norm** — per expert: ‖∂L/∂W_e‖ averaged over the batch.
   High = still learning (grow), ~0 = saturated or dead (shrink).
3. **Loss delta** — the held-out loss change if the expert were
   removed (measured by a probe pass, or approximated by
   `ΔL ≈ −‖g_e‖² / n` from the gradient — the standard quadratic
   approximation). Negative delta = the cell hurts (shrink).
4. **Route entropy** — the router's softmax entropy per expert.
   Near-0 entropy = the cell always wins or never wins (both are
   signals). This catches the "all tokens to one expert" collapse.

## The fitness signal (what the verifier accepts)

A mutation is **accepted** only if, after validation on a held-out
probe:
- **loss ↓** (or within tolerance) AND
- **the Lean proofs still pass** (wubu_prover2: Möbius closure,
  exp∘log, gyroassoc — the invariants) AND
- **the safety kernel holds** (no OOM, no NaN, route entropy not
  collapsed).

Accepted → archived (the DGM branch tree). Rejected → the 5+1
rollback restores the last healthy colony. This is the Darwin Gödel
loop: *empirical fitness, archive, open-ended, sandboxed.*

## The lifecycle (one epoch of the amoeba)

```
1. TRAIN: the colony learns (Muon + real backprop, the RC01 recipe).
2. DIAGNOSE: the immune system measures every cell (util, grad,
   loss-delta, entropy).
3. MUTATE: for each cell past a threshold -> grow or shrink; the
   others stasis.
4. VALIDATE: held-out loss + the prover + the safety invariants.
5. ARCHIVE or ROLLBACK: fitness up -> keep + archive; down ->
   the 5+1 slot restores the colony.
6. REPEAT: the colony is now a different size. The model can GROW
   from 35M to 70M and SHRINK back to 30M -- it adapts to the task,
   not to a fixed config.
```

## What we already have (the build list)

| Component | Status | File |
|---|---|---|
| the hive (the tissue) | DONE, tested | wubuwizard/src/wubu_hive.c |
| the mixed agents (the cells) | DONE, tested | wubuwizard/src/wubu_moe2.c |
| the prover (the fitness oracle) | DONE, tested | wubuwizard/src/wubu_prover2.c |
| the nesting (the space) | DONE, tested | wubuwizard/src/wubu_nest.c |
| the deltanet (the circulation) | DONE, tested | wubuwizard/src/wubu_deltanet.c |
| the real backward + Muon | IN PROGRESS | wubuwizard/src/wubu_backprop.c |
| the recovery (the safety) | DONE, on metal | wubuos/src/kernel/wubu_recovery.c |
| the trainer (the growth engine) | DONE | wubuwizard/src/wubu_train.c |
| **the diagnostic immune system** | DONE, tested | wubuwizard/src/wubu_amoeba.c |
| **the grow/shrink operators** | DONE, tested | same module |
| **the fitness gate** | DONE, tested | same module |

## The new module: wubu_amoeba

`wubu_amoeba` = the diagnostic immune system + the grow/shrink
operators + the fitness gate. It wraps the organs (hive, agents,
prover, trainer) and runs the lifecycle. Pure C11, opaque structs,
tested with finite-difference-style probes (the DA doctrine).

The API (from the header):
```
wubu_amoeba_init    — wire the organs
wubu_amoeba_train   — one training step (delegates to the trainer)
wubu_amoeba_diagnose— measure every cell (util/grad/loss-delta/entropy)
wubu_amoeba_mutate  — grow/shrink/stasis per the diagnosis
wubu_amoeba_validate— the fitness gate (loss + prover + invariants)
wubu_amoeba_commit  — archive the variant / rollback via the 5+1
wubu_amoeba_stats   — the colony's vitals (size, live cells, memory)
```

## The diagnostic model — one step further

"Diagnostic" has two meanings, and the amoeba honors both:
1. **The model diagnoses ITSELF** (the immune system above) — the
   model watches its own health and adapts.
2. **The model diagnoses THE WORLD** (the clinical meaning) — given
   an input, the colony's routing IS a diagnosis: the active cells
   are the "symptoms" it attended to. The route pattern
   (which experts fired, in what proportions) is WuBu's explanation
   of its own reasoning — the transparent, verifiable trace that
   the prestige ledger and the prover can check.

The two are the same loop: the amoeba diagnoses its inputs by
routing them through its cells, and diagnoses itself by watching
how those cells are used. **The route trace is the diagnosis; the
diagnosis is the model's self-knowledge.**

## The roadmap

1. **wubu_amoeba core** DONE (2026-08-03): the module + tests green, ASan-clean.
2. Wire it to the trainer (the colony trains between mutations).
3. Wire it to the recovery (the 5+1 rollback for bad mutations).
4. The archive (the DGM branch tree: every variant, its fitness,
   its parent — the lineage ledger).
5. Metal: the colony runs in ring-0 (the Live Colonel hosts it);
   the diagnostic loop is the AGI supervisor's heartbeat.
