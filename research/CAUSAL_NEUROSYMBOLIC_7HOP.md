# Causal + Neuro-Symbolic + Temporal — 7-hop Kevin-Bacon lily-pad KB sweep
## What AGI-at-home is STILL missing: the reasoning substrate above the vectors

> Each stone seeds the next hop. Target: map causal/neuro-symbolic/temporal
> findings to WuBuOS substrate gaps and close the tractable ones as C11.

## Hop 1 → Causal inference (SCM, do-calculus, discovery, counterfactuals)
Sources: ML beyond curve fitting (inference.vc), causal discovery (Meier 2025),
evaluating causality 2026 (futureagi.com).

Key findings:
- Structural Causal Model (SCM): variables + directed edges + noise. Induces
  observational p(x) and interventional p(x|do(a)) distributions.
- do-calculus: rules to push interventions through the graph; if a query is
  non-identifiable, no amount of data lets you estimate it.
- Causal discovery: PC, FCI, GES, Bayesian nets — learn structure from data.
  Requires assumptions (no missing nodes, correct arrow directions).
- Counterfactual: "what would have happened if..." — needs the SCM + noise.
- Three families: causal discovery, causal effect estimation, counterfactual.
- LLMs are correlation engines; they cannot do interventions without a causal
  model. This is THE missing piece for real agency.

WuBuOS gap: `wubu_worldmodel.c` (AG-04) does closed-loop verify-replan but has
NO causal structure — it predicts s' = A*s + b (a dynamical system, not a
causal graph). No do-calculus, no counterfactual, no identifiability check.

## Hop 2 → Neuro-symbolic (AlphaGeometry/Proof, KG+LLM, Engram, program synthesis)
Sources: neuro-symbolic 2026 turning point, zylos.ai agent reasoning, handbook
on neurosymbolic AI + KGs, DeepSeek Engram.

Key findings:
- Hybrid: neural for perception/pattern, symbolic for constraint/logic/
  auditability. Neither alone suffices.
- AlphaGeometry/AlphaProof: neural proposes, symbolic verifier checks. Can't
  hallucinate — wrong proofs rejected.
- KG + LLM: grounding language in structured facts reduces hallucination.
- DeepSeek Engram: router decides lookup (symbolic, O(1)) vs think (neural).
  Optimal split 75-80% compute / 20-25% memory.
- Program synthesis: LLM generates code; code executed + verified (transparent
  trace). SymCode (arXiv 2510.25975).
- 2026 = "Year of Neuro-Symbolic AI" driven by hallucination cost.

WuBuOS gap: no symbolic verifier in the decode path. The safety kernel
(wubu_safekern) is imperative, not logical. No KG, no rule engine, no
abductive diagnosis, no neuro-symbolic router.

## Hop 3 → Temporal reasoning / belief revision (PyReason, dynamic KG, Bayesian)
Sources: PyReason (temporal logic reasoning), TKG-Thinker (2602.05818),
EVOKG temporal KG, temporal reasoning over evolving KGs (2509.15464).

Key findings:
- PyReason: open-world temporal logic reasoning over KGs. Facts have time
  intervals; rules fire when conditions hold over time.
- Dynamic/temporal KGs: entities + relations + timestamps. Update from raw
  text (EVOKG). Multi-hop temporal inference.
- Belief revision: when new evidence arrives, update beliefs (Bayesian update).
  Handle contradictory evidence gracefully.
- EvoReasoner: temporal grounding + multi-hop inference under ambiguity.

WuBuOS gap: `wubu_agentic_mem.c` has episodic→semantic consolidation but no
temporal dimension, no belief revision, no timestamped facts, no dynamic KG.

## Hop 4 → Symbolic logic engine (Prolog, ASP, DL/OWL, description logics)
Sources: Prolog role in LLM era, OWL→Prolog translation, ASP+DL combination,
KR04 combining ASP with description logics.

Key findings:
- Prolog: first-order logic, relations, recursion, automated theorem proving.
  Excellent for symbolic reasoning + knowledge representation.
- ASP (Answer Set Programming): fully declarative; handles non-monotonic
  reasoning (defaults, exceptions). Prolog+ASP under one roof (Balduccini).
- Description logics (OWL Lite/DL): class hierarchies, restrictions.
  ASP+DL = rules + ontologies for the semantic web.
- A small Prolog/ASP engine is pure C, CPU-only, no GPU. Perfect for at-home.

WuBuOS gap: no logic engine at all. The system has no way to do deductive
inference over facts. This is the "symbolic" half of neuro-symbolic.

## Hop 5 → Symbolic planning (PDDL, STRIPS, HTN, neuro-symbolic planning)
Sources: LLM-Flax (2604.26569), LOOP closed-loop planner (openreview),
PDDL-INSTRUCT (MIT), neuro-symbolic planning (Kwon ICRA2025).

Key findings:
- PDDL/STRIPS: standardized planning spec. States = propositions; actions =
  preconditions + effects. Planner searches for action sequence to goal.
- HTN: hierarchical task networks — decompose high-level tasks into subtasks.
- Neuro-symbolic planning: LLM proposes candidate actions (neural), symbolic
  planner verifies preconditions/effects (symbolic). LOOP: 85.8% success,
  learns causal KB from executions.
- PDDL-INSTRUCT: finetune LLM to emit PDDL; 64x improvement on planning
  accuracy vs baseline.
- The pattern: neural proposes, symbolic validates. Closed loop refines.

WuBuOS gap: `wubu_worldmodel.c` (AG-04) does verify-replan but no STRIPS/PDDL
planner. The "plan" is a single predicted next state, not an action sequence
to a goal. No precondition/effect validation.

## Hop 6 → Counterfactual / abductive reasoning (diagnosis, explanation)
Sources: abductive reasoning taxonomy (2604.08016), HypoDeduce (abductive
fault localization), tackling hallucination with abduction (Galitsky 2025),
counter-abduction (MDPI 2026).

Key findings:
- Abduction: given observation O, infer hypothesis H such that H explains O.
  Fallible, ampliative (goes beyond data).
- Counter-abduction: generate RIVAL explanations; the initial H is defeated if
  a competing H is better supported. Transforms narrative reasoning into
  competitive, evidence-driven process.
- Hallucination = failure of abductive reasoning (missing premises). Abductive
  verifier exposes invented premises.
- NeSTR (2026): neuro-symbolic TEMPORAL reasoning with structured constraints.

WuBuOS gap: no abductive diagnosis. When something fails (e.g. gen_text OOM,
agent loop diverges), the system has no mechanism to hypothesize WHY and test
rival explanations. No counter-abduction to defeat a fluent-but-wrong CoT.

## Hop 7 → Integration with the AGI-OS substrate
- Causal layer (hop 1) feeds the world-model (AG-04): instead of s'=A*s+b,
  use a causal graph with do() interventions + counterfactuals.
- Symbolic verifier (hops 2,4) feeds the safety kernel (wubu_safekern): safety
  invariants expressed as logical rules, checked by the engine.
- Temporal KG + belief revision (hop 3) feeds agentic memory (wubu_agentic_mem):
  timestamped facts, dynamic update, Bayesian belief revision.
- PDDL planner (hop 5) feeds the world-model replan (AG-04): generate action
  sequence to goal, validate preconditions.
- Abductive diagnosis (hop 6) feeds the loopguard (wubu_loopguard): when a
  loop diverges or OOMs, hypothesize causes + test rivals.
- Neuro-symbolic router (hop 2 Engram) feeds the decode path: route static
  lookups to symbolic memory, reasoning to neural.

## Synthesis: WuBuOS causal/neuro-symbolic gaps
1. No causal graph (SCM) — only a dynamical predictor (AG-04).
2. No do-calculus / intervention — can't estimate p(x|do(a)).
3. No counterfactual — can't ask "what if".
4. No identifiability check — may attempt non-identifiable queries.
5. No symbolic verifier in decode path — safety is imperative, not logical.
6. No temporal KG / belief revision — memory has no time dimension.
7. No logic engine (Prolog/ASP) — no deductive inference over facts.
8. No PDDL/STRIPS planner — replan is single-step, not goal-directed.
9. No abductive diagnosis — no hypothesis generation when things fail.
10. No counter-abduction — no rival-explanation defeat of wrong CoT.

## Action plan: close causal/neuro-symbolic gaps as C11
- wubu_causal.c: SCM (graph + do + counterfactual + identifiability),
  abductive diagnosis (hypotheses + counter-abduction), temporal belief
  revision (Bayesian update over timestamped facts), PDDL-lite planner
  (preconditions/effects, goal-directed search). Closes gaps 1,2,3,4,6,8,9,10.
- wubu_symbolic.c: Prolog-ish rule engine (facts + rules + resolution) +
  constraint checker (safety invariants as logical rules). Closes gaps 5,7.
- Integrate: wubu_worldmodel uses wubu_causal SCM; wubu_safekern uses
  wubu_symbolic rule engine; wubu_agentic_mem uses temporal belief revision;
  wubu_loopguard uses abductive diagnosis on divergence.
- CPU-closable: graphs, resolution, Bayesian update, PDDL search are all
  pure C, no GPU needed.
