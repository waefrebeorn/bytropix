# Missing Needs: What Agentic / AGI-OS Systems Still Lack (2026 research sweep)
## gap analysis for the WuBuOS recursive-improvement loop

> Synthesized from: agentic-AI failure modes (MIT/YouTube), layered agentic
> security framework (arXiv 2604.23338), OWASP Agentic Top 10 (ASI01-10, 2025),
> OWASP LLM Top 10 v2 (LLM01-10), causal/planning gaps (Frontiers/ICML 2026),
> agentic identity gaps (Strata 9-problems), world-modeling (arXiv 2604.22748).

## The 8 missing needs NOT yet in the WuBuOS substrate

| # | Missing need | Source | WuBuOS status |
|---|--------------|--------|---------------|
| N1 | **Runaway-loop guard** (max step-count + hard timeout; terminate recursive agentic loops) | OWASP LLM10 "Unbounded Consumption"; ASI08 cascading failures | ✗ absent in decode path (no loop-termination enforcement) |
| N2 | **Goal-hijack / injection defense** (control-plane vs data-plane separation; detect instruction-in-data) | ASI01, LLM01; context-mixing defense | ⚠ partial (capzero 9P) but no injection DETECTION |
| N3 | **Memory/context poisoning detection** (cross-session replay; poisoned episodic memory) | ASI06, L3×T3 research gap | ✗ absent |
| N4 | **Closed-loop deliberative planning** (verify world-state, replan; not open-loop) | Open-loop problem; world-modeling 2604.22748 | ✗ absent (pure reasoning, no verify-replan) |
| N5 | **Trajectory-level audit attribution** (per-action immutable log + attribution) | L7×T1-4 accountability gap | ⚠ partial (ledger) but not per-agent-action trace |
| N6 | **Tool-abuse / excessive-agency cap** (cap tool calls per step/agent) | LLM06, ASI02 | ✗ absent (capzero grants but no rate cap) |
| N7 | **Inter-agent message authentication** (prevent spoofing/tampering) | ASI07 insecure inter-agent comms | ✗ absent |
| N8 | **Just-in-time provisioning + HITL gating** (sensitive-action human approval) | ASI08/strata JIT + HITL | ✗ absent |

## Which to close now (tractable, CPU-closable C11, no third-party deps)
- N1 runaway-loop guard → `wubu_loopguard` (max_steps + deadline + terminate)
- N2 control/data-plane separation → `wubu_planediv` (tag each input as control vs data; reasoner cannot act on data-plane as instruction)
- N3 cross-session poisoning replay → `wubu_poisondetect` (hash episodic memory; flag divergence)
- N5 trajectory audit → `wubu_trajaudit` (append-only per-action record w/ agent id + nonce)
- N6 tool-abuse cap → extend capzero with per-agent rate counter
- N8 HITL gating → `wubu_hitl` (sensitivity threshold -> require external approval token)

N4 (world-model closed-loop) and N7 (crypto inter-agent auth) are deeper; seed as
`open` research gaps but close the tractable 6 above as primitives + operator wiring.

Seeded as theme AG in research/INDEX.md. Closing begins next turn.
