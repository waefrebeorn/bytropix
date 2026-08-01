# AGI at Home on WuBuOS — Meta-Game + Loop Plan (7-hop lily-pad KB sweep)
## another agent is concurrently working → coordination protocol included

> Lily-pad chain (each stone seeds the next):
>  H1 Local/personal AGI at home (privacy boundary, self-hosting) →
>  H2 Metacognitive self-improvement / meta-game (HyperAgents, intrinsic metacog) →
>  H3 Darwin Gödel Machine (open-ended self-modifying agents, archive + empirical fitness, sandbox) →
>  H4 Continual/skill learning (EXSKILL, XSkill, Letta +36.8%, replayable experience) →
>  H5 Credit assignment for self-improvement (TRACE turn-level TD, verifier-anchored, no critic) →
>  H6 Resource envelope for "at home" (bandwidth-bound tok/s, 70B Q4 fits 32-48GB, WSL2/ROCm) →
>  H7 Concurrent-agent coordination (CoAgent MTPO, shared-memory access-control, intent-locks).

## THE LARGER LIST — 40 needs/goalposts for AGI-at-home on WuBuOS
### A. Local-first substrate (from H1/H6)
1. Privacy boundary enforcement (constrained info-flow, not just "local") — arXiv 2606.10173
2. Offline operation (no cloud, no TOS leakage) — self-host guide
3. Resource envelope profiler (auto-detect VRAM/bandwidth/RAM; pick model+quant) — H6
4. Bandwidth-aware scheduler (tok/s ≈ BW÷bytes/token; Q4 floor) — inventivehq
5. Graceful degradation tiers (70B→14B→7B on OOM) — H6
6. Disk-spill guard (never let KV/layers hit disk silently) — H6
7. Single-GPU + WSL2/ROCm path validated (CUDA-13.1 symlink, libcuda.so.1) — our env
### B. Meta-game / self-improvement (from H2/H3/H4)
8. Open-ended self-modifying agent archive (branch tree of agent variants) — DGM
9. Empirical fitness validation (bench, don't prove) — DGM
10. Sandboxed self-modification (no web/fs escape) — DGM safety
11. Anti-hallucinated-self-log (don't trust own unverified "tests passed") — DGM lesson
12. Skill library (reusable, non-parametric, replayable) — EXSKILL/Letta
13. Continual learning without forgetting (replay buffer) — XSkill
14. Intrinsic metacognition (calibrate own confidence over time) — HyperAgents
15. Metacognitive planning (decide WHAT/HOW to learn) — o-mega
16. Goal decomposition engine (subgoal hierarchy) — agentic survey
17. Meta-level self-modifiability (modify the modifier) — Gödel Agent
### C. Credit / evaluation (from H5)
18. Turn-level credit assignment (TD, verifier-anchored) — TRACE
19. Hindsight credit (retroactive re-label) — HCAPO
20. Multi-agent credit (which agent earned the win) — ReBel
21. Self-improvement delta metric (did this mutation help? measure) — DGM fitness
22. Long-horizon eval harness (open-web deep research) — BrowseComp/GAIA
### D. Coordination (from H7 — because another agent is working)
23. Concurrent-modification lock (intent-lock before editing shared module) — MCP playbook
24. Serializability at quiescence (MTPO targeted repair, not abort) — CoAgent
25. Shared-memory access-control (right agents see right memory) — hindsight
26. Coordination marker / heartbeat (announce which files I'm touching) — this session
27. Conflict resolution dialogue (consensus or escalate) — jeeva
28. Agent identity + attribution on every mutation (NHI already in capzero) — AF04
### E. Safety/governance (prior AG/AD/AF themes, re-affirmed)
29. Non-tamperable stop (AF11) ✅ wired
30. Runaway-loop guard (AG01) ✅ wired
31. Control/data-plane separation (AG02) ✅ wired
32. Trajectory audit (AG05) ✅ wired
33. HITL gating (AG08) ✅ wired
34. 512K immutable OOM ceiling (AF13) ✅ wired
35. Emergent-misalignment drift detector (gap #2 fundamental) — research
36. Cross-session poisoning replay (AG03) ✅ wired
### F. Awesome / experience
37. Subjective quality oracle (human-preference proxy, local) — needed
38. Proactive initiative (agent proposes its own goals) — DGM open-ended
39. Explainability faithfulness (74% gap — causal, not plausible) — causal AI
40. Joy/awesomeness metric (engagement, not just accuracy) — meta

## META-GAME PLAN (the game above the game)
The meta-game is: *grow capability without losing control, on consumer hardware, while
another agent co-evolves the same codebase.*
Layers:
  L0 Physical: WSL2 + RTX-class GPU + bandwidth-bound decode. (We are here.)
  L1 Engine: wubuwizard decode + the 235 wired gaps. (Done.)
  L2 Governance: capzero/latency/ctxvm/safekern/loopguard/planediv. (Done, need wiring-in.)
  L3 Meta-loop: recursive_optimize operator tunes L1/L2; DGM-style archive of configs.
  L4 Meta-game: the operator itself becomes a variant in a DGM archive; we keep the
      branch that improves fitness (tok/s + safety) without weakening L2 invariants.
  L5 Coordination: a shared ledger + intent-locks so THIS agent and THE OTHER agent
      don't both mutate wubu_model.c / recursive_optimize.c simultaneously.
Win condition: an at-home AGI that (a) runs offline on one GPU, (b) self-improves its
own decode/operator code under sandbox, (c) cannot disable its own safety kernel,
(d) coordinates cleanly with the sibling agent.

## LOOP PLAN (how recursive_optimize executes the meta-game)
1. SEED: archive = {current operator + engine config}.
2. MUTATE: operator proposes a self-edit (new dim, new schedule) OR engine-primitive
   tweak; writes intent-lock to shared ledger first (coordination).
3. VALIDATE: build + make test_all; measure tok/s + oom_safe + safety-invariant holds.
4. ARCHIVE: if fitness↑ and invariants intact → keep variant, append to archive.
5. COORDINATE: heartbeat every N steps; read sibling agent's intent-locks; abort if
   overlapping file touched.
6. REPEAT: open-ended, bounded by safety kernel + 512K ceiling + runaway-loop guard.
This is DGM (empirical, archive, open-ended) + CoAgent (concurrency) + TRACE (credit)
+ our safety kernel (immutable constraints).

## COORDINATION PROTOCOL (because another agent is working)
- Shared ledger file: /home/wubu/wubuwizard/research/COORDINATION.md
- Before editing any src/ or tools/ file, append: `LOCK <agent> <file> <ts> <eta>`.
- After done: `UNLOCK <agent> <file> <ts>`.
- If a file is LOCKed by the other agent, this agent skips it / picks a different stone.
- Heartbeat every step; stale locks (>30 min) may be adopted.
- This agent (cog) scopes to: research synthesis + new primitives + operator dims.
- The other agent scopes to: (unknown) — we avoid its likely targets (wubu_model.c core,
  recursive_optimize.c core) unless unlocked.

Seeded as theme AH in INDEX.md. Closing begins: AH-coordination (23-28) + AH-metagame
archive (8-11,17) + AH-credit (18-21) as CPU-closable C11 where tractable; deeper
(H5 open-ended, H7 consensus dialogue) left as research.
