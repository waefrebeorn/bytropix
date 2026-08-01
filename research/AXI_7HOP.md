# Self-Improving Code + Sandboxed Execution + Embodied Tool-Use — 7-hop KB sweep
## BA axis: the agent's ability to write+run+verify+act in the world

> Each stone seeds the next hop. Target: map the self-improvement + sandbox +
> embodied-cognition substrate that AGI-at-home is STILL missing.

## Hop 1: LLM code generation + execution (SymCode, Code-as-Policies)
SymCode (arXiv 2510.25975) — LLM generates self-contained, verifiable code (Python/C)
from specs, executes it, uses deterministic interpreter feedback to self-correct.
Code-as-Policies (arXiv 2209.07753) — re-purpose code-writing LLMs to write robot policy
code from natural language. Policy code = functions/loops that process perception →
control. Key: hierarchical code-gen (recursively defining undefined functions).
At home: the agent writes C11, compiles, runs, reads output, debugs → rewrites.
This is the "operate" half beyond the DGM archive.

## Hop 2: Sandboxed execution (gVisor, Firecracker, seccomp, WASM)
Five isolation levels: Docker (runc) → seccomp-bpf allowlist → gVisor (user-space
kernel, ~200 syscalls) → Firecracker microVM (~125ms boot) → WASM sandbox (zero-kernel).
Key insight: agent code is untrusted → must sandbox. For AI agent execution at home:
seccomp-bpf allowlist is the pragmatic sweet spot (no root, no GPU, no network).
Sandlock pattern (Landlock + seccomp + seccomp-notif): no root, no cgroups, COW fork.
gVisor for stronger (Sentry intercepts syscalls, only ~70 reach host).
Firecracker for max isolation (hardware-enforced, each workload own kernel).

## Hop 3: Formal verification of generated code (SAFE, rocq-of-rust, Aeneas, Lean)
SAFE (arXiv 2410.15756) — automated proof generation for Rust code via self-evolution.
rocq-of-rust — translates Rust to Rocq proof certificates. Aeneas (Lean Together 2026)
— formal verification of Rust crypto in Lean. Key insight: type-check + invariant check
is the pragmatic home-scale formal verification (no full proof assistant needed).
At home: C11 type-safety + invariant assertion is the verification gate before exec.

## Hop 4: Tool-use protocols (MCP, function calling)
MCP (Model Context Protocol, Anthropic-led, open spec) — JSON-RPC 2.0 over stdio/SSE/HTTP.
Tools declared with name+description+JSON Schema input. Parallel tool calls, structured
output. Function calling (OpenAI) — simpler but vendor-locked. MCP wins for portability.
At home: our agent already uses function-calling — standardizing to MCP schema =
decoupling tools from the model/transport layer. The agent's own tool-use is its
interface to the world (read_file, terminal, write_file).

## Hop 5: Self-modifying safety (DGM empirical gate, regression, lineage)
DGM (arXiv 2505.22954) — self-improving system modifies its own code, empirically
validates each change via coding benchmarks. Safety: anti-fake-log gate (verified=1
only when tests pass + OOM-safe), regression test suite, lineage trace.
ReVeal (2026) — self-evolving code agents via reliable self-verification.
CoEvoSkills — self-evolving agent skills via co-evolutionary verification.
Agent sandbox FAQ: "no plain Docker for untrusted code; gVisor or microVM baseline."
At home: our DGM gate (AX01) is the empirical validation; regression runner prevents
regression — but the CODE must also be verified before the agent runs it.

## Hop 6: Program synthesis for the agent (spec→code→verify→exec)
HEURIGYM (ICLR 2026) — LLM crafts optimization programs end-to-end.
Apeiron — scalable agentic framework for full-lifecycle code generation.
EvoAgent — autonomous-evolving agent with MLLM+WM, self-planning/self-reflection.
Code-as-Policies: hierarchical code-gen, recursively defining undefined functions.
At home: the agent generates C11 source from a spec (e.g., "new KV eviction policy"),
compiles it, runs regression tests, then injects the verified code into the decode path.

## Hop 7: Integration with AGI-OS substrate
The self-modifying loop:
  1. Propose code change (spec → C11 source)        [AX05 synth]
  2. Compile + type-check + invariant check         [AX03 verify, research]
  3. Run in sandbox (seccomp allowlist)             [AX02 sandbox, research]
  4. Run regression tests (test_all must pass)       [AX01 DGM gate]
  5. Run code-exec smoke test (rc/oom/latency)      [AX07 codeexec]
  6. If all pass → commit + inject into decode path; if fail → revert + log  [AX06 evolve]
  7. Safekern checks capability token before exec   [AX08 sandbox_safekern]
  8. Loopguard tracks self-mod cycles for runaway   [AG-01 loopguard]
  9. World-model verifies observed behavior vs predict [AG-04 world-model]

Closed-loop: the agent modifies its own substrate, and the substrate monitors the
agent that modifies it. This is the missing "act in the world" + "verify before run"
+ "sandbox untrusted code" capability.

## Gap mapping to WuBuOS substrate
- AX09 → wubu_verify.c (C11 type-check + invariant assertion before exec)
- AX10 → wubu_codesynth.c (spec→C11 source generator, template-based)
- AX11 → wubu_evolve.c already exists (AX06) — extend with exec-verifier bridge
- AX12 → wubu_codeexec.c can already exists (AX07) — wire into loopguard

## Note
AX02 (seccomp-bpf) and AX03 (formal verification / proof assistant) require kernel-level
or proof-assistant-level work. The C11-closeable subset: spec→code synthesis (AX10),
type-check+invariant gating (AX09), and the exec-verifier→loopguard integration (AX12).
AX11 extends the existing evolve loop.
