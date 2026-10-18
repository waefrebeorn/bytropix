# JevAgentHarness — delegation economics for the WuBu decision plane

**Seed:** arXiv 2610.21412, *JevAgentHarness: Benchmarking, Deploying and Training
Decision Models That Share Control with LLMs* (Zhu, Feng, Han, Wang, Yang, Tang,
You, Ji — UIUC / Princeton / Google), submitted 15 Oct 2026.

**Wave:** 7 layers × 25 queries + 25 convergence queries = **200 queries,
1,000 records, 969 unique URLs**, layers 1–7 written to
`/home/wubu/opt/research/jev-harness-2026-10-18/`. Layer 7 and the convergence
pass are search hits, not citations; only the seed was read in full.

**Licence:** paper **CC BY-NC-SA 4.0**; StartLux-Decision weights **CC BY-NC
4.0**; the released harness/models are therefore **research-only**. Nothing from
this paper's artifacts may enter a training shard. Same discipline already
applied to Jev-RAG-130K (`eval_only`).

## The claim

A harness calls a large LLM at every step, but most calls end in a multiple-choice
decision (which tool, which argument, stop, reflect, route). A small decision
model answers those directly: in one recorded episode a Jev call cost **28% of an
LLM call's time and 2.6% of its price**. Delegation is safe only above a
per-decision-point confidence threshold; below it the LLM decides.

The break-even rule, when confidence equals accuracy:

```
tau_d = q_L - dC / V
```

`q_L` = how often the LLM gets that decision right, `dC` = price of the saved
call, `V` = cost of a wrong choice (drop in task success, valued at $1/task).
Thresholds are fitted per decision point on **shadow-mode trajectories** (150
tasks per agent LLM), not guessed.

## What they measured

| Finding | Number |
|---|---|
| Isolated accuracy misranks models | StartLux-9B: 92.2% BFCL tool choice → **69.5%** task success; AgentDecision-4B: 89.3% BFCL → **91.5%**; Jev 82.0% |
| A fixed threshold is not enough | threshold fixed at 0.5 loses **21.5 points** |
| Fitted cascade | 27.2% of tool choices + 20.8% of arguments delegated → **14.2% cheaper, zero tasks lost** (500 tasks) |
| Replace mode | **−19.0 points** on tool choice |
| Serving a 4B locally | **86.3% cheaper, 2.5× faster**; 99.9% of tools, 75% of args, 4.5 fewer LLM calls/task |
| Latency | 31 ms (4B, L4) vs 809 ms (Flash) per decision |
| Training | SFT on 587 tasks: step accuracy on shifted states **52.9% → 98.4%**; 2.2 GPU-h SFT + 4.5 GPU-h GRPO |

The must-ask states are where models separate: AgentDecision-4B 67.4% correct vs
Jev 58.3%. Every failed Jev task ended at its **first wrong tool choice**, usually
an irreversible action (a refund) at a state where asking was the only acceptable
move.

## The negative results (the useful half)

- Coding-agent step routing saved only **7.0–10.5%**, and moved success both ways
  by harness: pi +7.9 points (noise), SWE-agent **−6.0**.
- A proxy deciding "read-only vs writes/executes" separated poorly: **AUROC 0.52**
  in pi, 0.56 in mini-SWE-agent. No full run was justified.
- With Flash-Lite as the agent LLM the fitted threshold left **every** tool choice
  to the LLM, i.e. the cascade correctly saved nothing.

## What WuBu already had, and what this added

WuBu's `wubu_decision` module already enforced the arithmetic contract the hosted
models do not (exact softmax, defer band, complement/exclusivity, ECE). This
paper contributes the **economics**, which was missing. Added to
`include/wubu_decision.h` / `src/wubu_decision.c`:

- `wubu_decision_analytic_threshold(qL, dC, V)` — the closed-form break-even.
- `wubu_decision_fit_threshold(samples, n, qL, cost_probe, cost_llm, cost_wrong,
  budget)` — scans shadow-mode samples, returns the cheapest threshold whose
  projected success stays within the budget, `2.0` when delegating nothing wins.
- `wubu_decision_ledger` — delegated/deferred counts, wrong delegations, cost
  spent/avoided, time spent/avoided, and percent saved against an LLM-always
  baseline.

`tools/test_decision` now runs **31 checks, 0 failures**: analytic tau 0.89, fit
lands at 0.95 on a two-band fixture, catastrophic wrong-cost returns "delegate
nothing", and the ledger reproduces 151.0 cost / 24.5% saved exactly.

## The seven-hop convergence

1. **Jev (decision model)** — a probability per option instead of text.
2. **Router/cascade literature** (RouteLLM, FrugalGPT) — the same deferral shape,
   but on fixed questions, not agent states.
3. **Selective prediction / calibration** — confidence must be fitted, never
   assumed; ECE is the diagnostic.
4. **On-policy distillation** — train on states the agent actually reaches, not
   on a fixed set; that is where 52.9 → 98.4 comes from.
5. **GRPO with calibration-aware reward** — success *and* fewer LLM calls.
6. **Agent benchmarks (τ²-bench, ALFWorld, SWE-bench, Terminal-Bench)** — score
   whole tasks; they cannot say *which* choice failed.
7. **DecisionGym** — a scripted environment with an answer key at every reachable
   state, which is what makes step-level scoring possible at all.

Convergence: **a control plane is only as good as its answer key.** Every model
number depends on knowing the acceptable action at the state the agent is
actually in, including after its own mistakes. That is a world-model property,
not a model property.

## Lilypad actions for WuBu

1. **Done** — threshold fitter + ledger + 3 new invariants in `test_decision`.
2. **Must-ask detection in the OS decision plane.** WuBuOS has irreversible
   syscalls (unlink, revoke, format). The paper's failure mode is exactly a
   confident irreversible action at a must-ask state. The decision gate should
   carry a `must_ask` class that forces deferral regardless of confidence.
3. **Answer keys from the 9P namespace.** DecisionGym derives an answer key from
   a scripted plan; WuBuOS already has a namespace, so a task's acceptable next
   steps can be derived from the namespace state rather than hand-written.
4. **Jev-RAG-130K as a calibration fixture.** 130,000 claim decisions with 21,181
   teacher/reference disagreements give real confidence/accuracy pairs; feed them
   to `wubu_decision_fit_threshold` and report the fitted tau per task family —
   research-only, never training.
5. **Do not chase coding-agent routing.** The paper's own numbers say 7–10.5% at
   best and negative in SWE-agent. Spend the effort on the OS control plane,
   where the decisions are structural and the answer key is derivable.

## Honest caveats

- One seed paper read in full; layers 1–7 are discovery, not verified citations.
- The released artifacts are non-commercial; any WuBu experiment must stay
  research-only and must not contaminate training shards.
- The 3-point success budget is a policy choice, not a law. WuBu's fitter accepts
  it as a parameter for that reason.
