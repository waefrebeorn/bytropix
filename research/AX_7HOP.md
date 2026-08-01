# Self-Improving Code + Sandboxed Execution + Verifiable Tool-Use — 7-hop KB sweep
## AX axis: the agent's ability to write+run+verify its own code safely

> Each stone seeds the next hop. Target: map the self-improvement + sandbox + tool-use
> substrate that AGI-at-home is STILL missing.

## Hop 1: LLM code generation + execution (SymCode, code-as-reasoning)
SymCode (arXiv 2510.25975) — LLM generates self-contained Python scripts, executes them,
uses deterministic interpreter feedback to self-correct. Key insight: code is a transparent
reasoning trace. The LLM writes code → runs it → reads output → debugs → rewrites.
At home: this is the "agent writes C11, compiles, runs, reads output" loop.

## Hop 2: Sandboxed execution (gVisor, Firecracker, seccomp, WASM)
Five isolation levels: Docker (runc) → seccomp → gVisor (user-space kernel) →
Firecracker microVM → WASM sandbox. For agent code at home: seccomp-bpf allowlist
is the pragmatic sweet spot (no root, no GPU, no network). gVisor for stronger isolation.
Firecracker for maximum security. WASM for zero-Kernel sandboxed compute.

## Hop 3: Formal verification of generated code (SAFE, rocq-of-rust, Aeneas, Lean)
SAFE (arXiv) — automated proof generation for Rust code via self-evolution.
rocq-of-rust — translates Rust to Rocq proof certificates. Aeneas (Lean Together 2026)
— formal verification of Rust crypto in Lean. Key insight: type-checking + invariant
checking is the pragmatic home-scale formal verification (no full proof assistant needed).

## Hop 4: Tool-use protocols (MCP, function calling, JSON Schema)
MCP (Model Context Protocol, Anthropic-led, open spec) — JSON-RPC 2.0 over stdio/SSE/HTTP.
Tools declared with name+description+JSON Schema input. Parallel tool calls supported.
Function calling (OpenAI) — simpler but vendor-locked. MCP wins for vendor-neutral portability.
At home: our tool-use is already via function-calling in gen_text; MCP compatibility =
standardizing the schema.

## Hop 5: Self-modifying safety (DGM empirical gate, regression testing)
DGM (Darwin Gödel Machine) — self-improving system modifies its own code, empirically
validates each change via coding benchmarks. Safety: anti-fake-log gate (verified=1 only
when gen_text returns 0 AND oom_safe), regression test suite, lineage trace.
ReVeal (2026) — self-evolving code agents via reliable self-verification.
CoEvoSkills — self-evolving agent skills via co-evolutionary verification.

## Hop 6: Program synthesis for the OS (config/plan generation)
Agentic OS patterns (2026) — LLM generates operational plans, config files, system
specifications. HEURIGYM — agentic benchmark for LLM-crafted optimization programs.
Apeiron — scalable agentic framework for full-lifecycle code generation.
At home: our recursive_optimize already does plan generation; this gap is about
generating C11 source code (not just config) with verified compilation.

## Hop 7: Integration with AGI-OS substrate (code-exec verifier → loopguard, sandbox → safekern)
The self-modifying loop: propose code change → compile → run regression tests →
verify safety invariants → if pass, commit; if fail, revert + log.
Code exec verifier feeds loopguard (runaway-loop detection on self-mod cycles).
Sandbox feeds safekern (capability/zero-trust on exec'd code).
DGM archive records every code variant with verified flag.
CoAgent coordination ensures only one agent modifies code at a time.

## Gap mapping to WuBuOS substrate
- AX01 → wubu_dgm.c (DGM empirical gate + regression test runner)
- AX02 → wubu_sandbox.c (seccomp-bpf allowlist + namespaces)
- AX03 → wubu_verify.c (type-check + invariant check stub)
- AX04 → wubu_tooluse.c (MCP-compatible tool schema + dispatch)
- AX05 → wubu_synth.c (program synthesis: spec→C11 code gen)
- AX06 → wubu_evolve.c (self-evolution loop: propose→verify→commit→regress)
- AX07 → wubu_codeexec.c (code exec verifier → feeds loopguard)
- AX08 → wubu_sandbox_safekern.c (sandbox capability bridge → safekern)

> Note: AX02 (seccomp-bpf) and AX03 (formal verification) require kernel-level
> or proof-assistant-level work that exceeds a single C11 module. These are marked
> as `research` gaps — the tractable subset (AX01, AX04-AX08) closes as C11.
EOF