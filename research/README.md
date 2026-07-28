# bytropix Research Archive (Kevin-Bacon convergence store)

This directory is the **persistent, cross-session research document system** for the
bytropix C/C++ LLM inference engine. Every optimization idea that survives the
triple-DA audit gets a numbered document here so we can re-derive it, re-audit it,
and re-implement it without re-searching the web.

## How a gap becomes a doc
1. Online search (Kevin-Bacon 7-hop, large keyword base, newest-first).
2. **Triple-DA audit** each candidate:
   - Pass 1 CORRECTNESS: is the math/algo sound? does it apply to our decode-bound,
     CPU/own-kernel, no-Triton, 13 GB RAM reality?
   - Pass 2 PRIVACY/SAFETY: does it need external services, telemetry, or
     weight downloads we don't control? (we reject anything requiring 3rd-party libs)
   - Pass 3 ROBUSTNESS: does it degrade gracefully? any numerical-stability cliff?
3. If it passes, write `NNN-slug.md` with: source(s), core idea, why it converges
   with the other findings, the OWN-C-implementation plan, and a test oracle.
4. Cross-link in `INDEX.md` under a convergence theme.

## Convergence themes (so far)
- **MEMORY-BANDWIDTH-BOUND DECODE** — the spine. Roofline (2607.02558) says
  decode GEMV + KV read are BW-bound. Every win below attacks bytes moved:
  KV quant (Q8_0/KIVI/Ecco entropy), weight quant (int4 Marlin / int8 GEMV /
  BitNet 1.58 / SmoothQuant), multi-tier KV (DRAM/NVMe/CXL/IB), near-memory compute.
- **STRUCTURE-OF-ARRAYS / CACHE-AWARE LAYOUT** — console-game data-oriented
  design: SoA, arena allocators, fixed-timestep, cache-line packing. Maps directly
  onto tensor storage, KV pages, expert routing tables.
- **DISAGGREGATED PREFILL/DECODE + KV TRANSPORT** — Mooncake/DistServe/Dynamo/
  LMCache/NIXL. On a single-host CPU engine this becomes: separate prefill pass
  that writes KV, then a decode pass; KV reuse across requests (prefix cache).
- **ARCHITECTURE VARIANTS** — GQA/MQA/MLA, Gated-DeltaNet hybrids (3:1),
  Mixture-of-Depths, fine-grained MoE. These are *model* properties we must
  support in the loader/forward, not engine hacks.
- **FORMAL VERIFICATION OF KERNELS** — Alive2 / bounded translation validation /
  equivalence checking. Use to *prove* our quantized GEMV == reference, not just
  cosine-test.

## Loop discipline (see skill: kevin-bacon-research-audit)
Search → audit → doc → implement → test → re-audit. The skill re-reads INDEX.md
so each session continues the same grind instead of restarting.
