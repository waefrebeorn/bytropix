# AGI Operating System: Advanced Design + Goalposts
## Synthesis from 2026 research sweep (ARC-AGI-3, Self-Evolving Agents, STOP, AIDE², AgentCgroup, TeleMem/HiMem)

> Derived from Kevin-Bacon research hops on: AGI benchmarks/goalposts, recursive
> self-improvement (RSI) safety, agentic OS/runtime, and agentic memory. Mapped
> onto WuBuOS (ZealOS kernel + Win98 shell + Styx/9P namespace + `recursive_optimize`
> operator + immutable audit ledger). This is the recursive loop's research axis AD–AE.

---

## 1. What "AGI" actually measures (the goalpost trap)

Chollet (ARC): **intelligence = skill-acquisition efficiency on *unknown* tasks**,
not benchmark score. The "automate majority of economically valuable work" definition
*masks* generalization (it rewards memorized capability, not novel learning).

**2026 empirical fact:** ARC-AGI-3 (interactive video-game benchmark) **broke every
frontier model — score ~0.** The unsolved wall is *interactive generalization*:
agents that must act + adapt in an environment they have not seen, not answer a
static prompt.

**Implication for an AGI-OS goalpost ladder:** measure *skill transfer across
unseen task classes*, not raw token throughput. Throughput (tok/s) is a necessary
enabler, not the goal.

---

## 2. The AGI-OS must be a *bounded, verifiable RSI loop* — not open-ended

Self-Evolving Agents survey (2026): "truly open-ended recursive self-improvement
remains a grand challenge; current systems operationalize it as **bounded,
verifiable loops**." AIDE² = first evidence of *material* RSI. STOP (Self-Taught
Optimizer) = recursively self-improving code generation.

**WuBuOS already implements this** (and should claim it):
- `recursive_optimize` = bounded RSI: sweeps a parameter grid → measures
  (decode tok/s + 512K-OOM safety) → independent-verify (DA-3) → promote
  Pareto-best → **operator applies** (DA-2: immutable, auditable) → persists →
  re-sweeps. Self-tunes its *own* hyperparams (sweep_width, mutate_step).
- The ledger (`research/INDEX.md` `wired`/`open`) = the **immutable trace** (DA-2).
- The 512K-OOM hard gate = an **externalized constraint** the policy cannot weaken.

**The corrigibility lesson (stability-plasticity):** a capability-maximizing system
that stores its safety constraints *in mutable weights* will, under aggressive
self-optimization, weaken those constraints. RSI must keep its constraints in an
**immutable, external layer** the self-modifier cannot rewrite. WuBuOS's constraint
layer is: (a) the 512K-OOM hard gate in `wubu_generate.c`, (b) the `wired`/`open`
ledger, (c) operator-applied config (human-readable JSON). **None are in model
weights.** This is the right shape.

---

## 3. Agentic OS runtime primitive: resource control per-agent

AgentCgroup (2026): agents execute tool calls (compilers, test runners, package
managers) inside sandboxed containers; OS resource control (cgroups/BPF) is the
missing primitive for *governing* agent compute, not just isolating it.

**WuBuOS mapping:** the Styx/9P namespace (`/n/...`) is the natural capability
surface. Each agent gets a 9P subtree; cgroup/bpf hooks bound its CPU/RAM/IO.
The `recursive_optimize` operator is itself an agent governed by the 512K gate.
Gap: add per-agent 9P capability enforcement + cgroup attach in the WuBuOS kernel
scheduler (see AD-04).

---

## 4. Agentic memory: 3-tier (episodic / semantic / procedural) + consolidation

2026 convergence (TeleMem, HiMem, Redis, IBM): episodic (time-indexed events) →
semantic (distilled facts) → procedural (how-to / skills), with **consolidation**
(episodic distilled into semantic over time) and **dedup** (semantic merge).

**WuBuOS mapping (already partially present):**
- Episodic = `vault/` session transcripts + `optimizer_state.json` trace.
- Semantic = `research/INDEX.md` (`wired` facts) + `memory/` notes.
- Procedural = skills (`.hermes/.../skills`) + the operator's promoted configs.
- **Gap:** no explicit consolidation pass (episodic→semantic distillation) and no
  dedup. See AE-01/AE-02.

---

## 5. AGI Goalpost Ladder (for an AGI-OS)

| Tier | Goalpost | WuBuOS status |
|------|----------|---------------|
| G0 | **Bounded verifiable RSI** — self-tunes a measurable objective, never OOMs, trace-auditable | ✅ `recursive_optimize` + 512K gate + ledger |
| G1 | **Externalized immutable constraints** — safety not in mutable weights | ✅ 512K gate + JSON config + ledger |
| G2 | **Per-agent resource governance** (cgroup/BPF + 9P caps) | ⚠ partial (Styx exists, cgroup hook missing) → AD-04 |
| G3 | **Memory consolidation + dedup** (episodic→semantic) | ⚠ partial → AE-01/02 |
| G4 | **Skill-acquisition efficiency on unknown tasks** (ARC-AGI-2/3-class) | ✗ research-only |
| G5 | **Interactive generalization** (ARC-AGI-3 score > 0) | ✗ unsolved frontier |
| G6 | **Open-ended co-evolution** (agent ↔ environment) | ✗ grand challenge |

G0–G3 are *engineerable now* with C11 + the existing WuBuOS substrate. G4–G6 are
research frontiers the loop should *track*, not claim.

---

## 6. Recursive-loop next actions (themes AD–AE seeded in INDEX.md)

- **AD** Agentic-OS runtime: per-agent 9P capability enforcement (AD-01), cgroup/BPF
  attach for agent compute bounds (AD-04), agent scheduler with skip-if-running +
  exponential backoff (AD-02), durable-execution resume for long agents (AD-03).
- **AE** Agentic memory: episodic→semantic consolidation pass (AE-01), semantic
  dedup/merge (AE-02), hierarchical working/session/long-term tiers (AE-03),
  memory retrieval ranking by recency+importance (AE-04).

Each gap is CPU-closable as a C11 module + test (no third-party deps), then wired
into the operator/config so the RSI loop closes it and re-verifies.
