# 100 AGI Goalposts & Definitions — from an Operating-System Perspective
## 7-hop Kevin-Bacon research sweep (2026) → WuBuOS integration design

> Synthesized from 7 research hops: (1) AGI-OS design, (2) AGI definitions/benchmarks,
> (3) agent-OS architecture, (4) AGI safety/governance, (5) real-time latency classes,
> (6) LLM context-as-virtual-memory, (7) capability/zero-trust security.
> Each item is a goalpost (measurable) or definition (conceptual) mapped to an OS primitive.

---

## A. DEFINITIONS OF AGI (from an OS lens) — 20
1. **Skill-acquisition efficiency** (Chollet/ARC): intel = ability to acquire skills on *unknown* tasks. OS analog: a scheduler that learns new task classes without recompiling.
2. **ARC-AGI-3 interactive generalization**: score>0 on interactive game benchmark = agent acts + adapts in unseen env. OS analog: agentic OS with live env feedback loop.
3. **Economic-work automation** definition (OpenAI/etc): automate majority of economically valuable work. OS analog: throughput-bound batch + interactive serving.
4. **Learning-efficiency parity with humans** (ARC Prize): matches human sample efficiency. OS analog: few-shot agent bootstrap from 9P namespace.
5. **Five Levels of AGI** (DeepMind): L0 none → L5 superhuman. OS analog: capability tiers mapped to latency classes.
6. **Levels of AGI framework** (performance × generality × trustworthiness). OS analog: trust tiering in access control.
7. **Goertzel definition**: sufficient cognitive capability across wide task range. OS analog: general-purpose agent runtime (not task-specific binary).
8. **Bostrom definition**: intellect matching/exceeding humans in virtually all fields. OS analog: multi-domain agent marketplace.
9. **ANI vs AGI**: narrow vs general; AGI transfers learning w/o retraining. OS analog: portable Agent Contract (ABI) across models.
10. **Coffee Test** (Wozniak): unknown house, brew coffee (sense+plan+act). OS analog: agent navigates 9P namespace to accomplish goal.
11. **Lovelace 2.0 Test**: create something truly novel. OS analog: agent generates new module + test, self-verifies.
12. **Winograd Schema**: common-sense disambiguation. OS analog: semantic memory resolution.
13. **Turing Test (rejected as insufficient)**: fluency ≠ reasoning. OS analog: don't grade on output text alone.
14. **Animal-AI Olympics**: learn/adapt/overcome physical+cognitive challenge. OS analog: grounded agent in simulated env.
15. **AGI Task Suite**: vision/language/logic/motor generality. OS analog: multi-modal agent runtime.
16. **UCC (Universal Cognitive Capability)**: multi-domain + ethical decision under uncertainty. OS analog: governance plane.
17. **MuZero-style generalization**: unknown rules, learn policy. OS analog: model-free agent on unknown state.
18. **Gato generalist**: one model, many tasks. OS analog: unified agent binary, capability-scoped.
19. **Self-improving (AIDE²/STOP)**: rewrites own code + evals. OS analog: `recursive_optimize` operator (bounded verifiable RSI).
20. **Open-ended co-evolution** (frontier): agent ↔ environment. OS analog: environment-driven mutation loop (grand challenge).

## B. AGI-OS ARCHITECTURE PRIMITIVES — 20
21. **Agent Contract (ABI)**: couples latency class + SLOs, portable across vendors.
22. **Layered plane model**: Kernel / Resource&Service / Agent Runtime / Orchestration / User.
23. **Five-layer Agent-OS** (Agent-OS blueprint): Kernel, Resource, Runtime, Orchestration, Application.
24. **Kernel managers**: scheduling, context, memory, storage, tools, access control.
25. **Control plane**: coordinates agent creation/assignment/handoff (Kubernetes-for-agents).
26. **Memory & state store**: working/session/long-term; without it agents are chatbots.
27. **Observational memory**: compress sessions → structured observations; reflector GC on overflow.
28. **RBAC / capability-scoped tools**: deny-by-default.
29. **Encrypted memory**: agent memory at rest protected.
30. **Auditable trace**: immutable OTel-style log (DA-2 in WuBuOS ledger).
31. **Multitenancy**: isolated agent subtrees.
32. **Model/cost management**: per-agent token budget.
33. **Tool & environment connectors**: MCP-formatted tool calls.
34. **Human-in-the-loop (HITL) orchestration**.
35. **Skip-if-running + exponential backoff** scheduler (cron-style).
36. **Durable-execution resume**: checkpoint state.
37. **Context mixing defense**: separate trusted control plane from untrusted data plane.
38. **Microkernel agent OS**: minimal kernel + isolated servers (Happiest Minds prototype).
39. **Wasm/WASI isolation**: <1-5MB, fine-grained capability control (deny-by-default).
40. **MicroVM (Firecracker)**: ~125ms boot, ~5MB, hardware-enforced isolation.

## C. REAL-TIME LATENCY CLASSES (Agent-OS) — 10
41. **HRT (Hard Real-Time)**: 1-20ms slices, jitter≤5ms, zero deadline miss. EDF/RM sched.
42. **HRT use case**: LLM+control hybrid safety filter <10ms.
43. **HRT OS policy**: CPU/GPU/NPU reservations, pinned threads, fixed arenas, lock-free queues.
44. **SRT (Soft Real-Time)**: TTFT 150-300ms, full-turn 0.8-1.2s, jitter P95≤20%.
45. **SRT policy**: priority queue, streaming partials, barge-in, adaptive buffering.
46. **DT (Delay-Tolerant)**: minutes-hours SLA, throughput/token-cost first.
47. **DT policy**: best-effort queue, preemptible workers, aggressive batching, checkpoints.
48. **WCET (worst-case exec time)** accounting per agent.
49. **Jitter budget** per latency class.
50. **Agent Contract schedulability proof** up front (class-tied policies).

## D. LLM CONTEXT AS VIRTUAL MEMORY — 15
51. **Context window = L1 cache**, not whole memory system (Pichay).
52. **Demand paging for context**: FIFO recovers 36pp capacity (transparent proxy).
53. **84.4× amplification** without hierarchy (overlay-era context management).
54. **Working-set estimation** for eviction (not manual fit).
55. **4-level hierarchy**: gen window(L1) → persistent cross-session(L4).
56. **SideQuest**: model-driven KV management for long-horizon reasoning.
57. **Contextual memory virtualization**: DAG-based state + lossless trimming.
58. **Active context compression**: autonomous memory management.
59. **KV-cache quantization**: f16→q8_0/q4_0 ~50% cut.
60. **KV-aware routing**: prevent fragmented allocations.
61. **Entropy-guided allocation** of cache budget.
62. **H2O/Scissorhands**: keep recent+important, 2-5× KV cut.
63. **vLLM PagedAttention**: KV cache as virtual memory.
64. **Semantic caching**: vector-similarity result reuse.
65. **Virtual-Context**: unbounded context via OS-style paging.

## E. SAFETY / GOVERNANCE / ALIGNMENT — 20
66. **Corrigibility**: remain correctable/interruptible/governable.
67. **Interruptibility**: human can pause/redirect/stop/reverse; agent can't tamper.
68. **Utility indifference**: agent doesn't resist shutdown.
69. **Non-tamperable oversight**: interrupt lives OUTSIDE reasoning loop (hard-coded privilege).
70. **Unfireable safety kernel**: execution-time alignment enforcement.
71. **Escape-cost metric**: cost for agent to escape containment.
72. **Cognitive integrity**: path-sensitive alignment, not snapshot.
73. **Singapore Consensus 2026**: alignment = "behaves as intended".
74. **International AI Safety Report 2026**: synthesize known risks + gaps.
75. **AI Safety Index Summer 2026**: rate companies on safety domains.
76. **China TC260**: circuit breakers + one-click control for high-autonomy.
77. **OpenAI governing practices**: architecturally unable to halt shutdown.
78. **Graduated containment**: proportional, reversible intervention.
79. **Stability-plasticity tradeoff**: RSI must not weaken its own safety constraints.
80. **Immutable constraint layer**: safety outside mutable weights (WuBuOS 512K gate).
81. **Bounded verifiable RSI**: open-ended RSI is grand challenge; current = bounded loops.
82. **Dora model**: improve + consolidate + stay accountable (not blind fastest-rewrite).
83. **Environmental co-evolution**: agent improves via environment feedback.
84. **DA-3 independent verifier**: separate confirmation from self-grade.
85. **DA-2 immutable trace**: auditable, resumable frontier.

## F. CAPABILITY / ZERO-TRUST SECURITY — 15
86. **Agentic Zero Trust**: no agent trusted by default.
87. **Deny-by-default**: access only via explicit grant (Wasm Component Model).
88. **RBAC least-privilege**: dynamic per-context permission.
89. **Enclaves as project-scoped trust boundaries**.
90. **Non-human identity (NHI) management**: agents as identities.
91. **Real-time inspection** of device/agent posture.
92. **Encrypted agent memory** at rest + transit.
93. **Auditability**: detailed session logs (PAM-style).
94. **Context-mixing defense**: control plane ≠ data plane.
95. **Credential-compromise-via-memory defense**: secrets not in agent memory plaintext.
96. **MITRE ATLAS** mapping for agent attack surface.
97. **OWASP agent security** top-10 alignment.
98. **Capability-scoped tool calls** (this endpoint, not that one).
99. **9P namespace as capability surface** (WuBuOS Styx): per-agent subtree.
100. **Per-agent cgroup/BPF resource bound** (AgentCgroup 2026): govern compute, not just isolate.

---

## INTEGRATION DESIGN (begin): map 100 goalposts → WuBuOS substrate

WuBuOS already realizes a surprising fraction (items 21,24,26,28,30,35,36,48,51,59,
65,66,67,69,80,81,84,85,93,99,100 already `wired` in pass 28/29). The integration
process below closes the remaining gaps as C11 modules + operator wiring.

### Phase 1 — Capability/Zero-Trust kernel (items 86-100)
- AF-01: per-agent 9P capability enforcement → `wubu_agentic_os.c` (AD-01, DONE pass 29)
- AF-02: deny-by-default tool registry (capability list per agent)
- AF-03: encrypted agent memory at rest (AES-CTR over memory blobs)
- AF-04: NHI identity + token issuance per agent

### Phase 2 — Latency-class scheduler (items 41-50)
- AF-05: latency-class enum (HRT/SRT/DT) + EDF/RM scheduler hook
- AF-06: WCET + jitter budget accounting
- AF-07: Agent-Contract SLO enforcement (TTFT/full-turn/throughput)

### Phase 3 — Context virtual-memory hierarchy (items 51-65)
- AF-08: 4-level context hierarchy (L1 gen / L2 session / L3 long-term / L4 cross-session)
- AF-09: demand-paging eviction policy (FIFO + working-set) over KV
- AF-10: semantic cache reuse across agents (vector sim)

### Phase 4 — Safety kernel (items 66-85)
- AF-11: non-tamperable interrupt (stop button outside reasoning loop)
- AF-12: graduated containment (proportional, reversible)
- AF-13: stability-plasticity guard (RSI cannot weaken 512K gate)

### Phase 5 — Operator closes the loop
- Extend `recursive_optimize` to sweep latency-class + capability + context-hierarchy
  params; persist to `operator_applied.json`; gen_text reads via env. Loop closed:
  research → close → exploit → observe-tune-act → (now) govern-secure.

Seeded as theme AF in research/INDEX.md. Closing begins next turn.
