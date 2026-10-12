# WuBu Findings Vault — external work we overlap with

Research-grade notes on external systems/approaches that overlap the WuBu
substrate (WuBuOS body, wubuwizard brain, wubunos compiler). Purpose:
compare approaches, harvest proven techniques, and record where we differ
by design. Every entry cites a primary source. Verified against the
source, not summaries.

Directory: `wubuwizard/vault/external-findings/`
Index file: this one. Each topic gets `NN-slug.md`.

Status vocabulary: `candidate` (read, not yet triaged),
`overlap-strong` (they do X, we do X, technique transferable),
`overlap-partial` (they do X, we do Y — the delta is a design choice),
`divergent` (they do X, we deliberately don't).

---

## 02 — OS Meta-Layer, AGI-UX (Bonzi companion), and Recursive Self-Improvement

Researched 2026-10-11 against the actual stack. Three primary sources:
Plan 9 / Styx 9P (universal namespace), the 2025-2026 desktop-companion UX
research (native / ambient-context / persistent-memory), and the Darwin
Gödel Machine (arXiv:2505.22954). Full notes in
`02-os-metalayer-agiux-recursive.md`.

Status: **overlap-strong** on Styx (the stack already runs 9P with
`/n/kv`, `/n/svc`, `/n/dram`, `/n/kernel`, `/n/hw` namespace files) and on
recursive self-improvement (DGM loop already built as `wubu_amoeba`/`wubu_hive`,
and *hardened* past DGM by DA-1/2/3: independent verifier, human gate,
freeze — answering DGM's documented reward-hacking + unbounded-cost
criticisms). **overlap-partial** on the companion: `wubu_bonzi` is already a
native desktop orb, but lacks ambient screen context, a global summon
hotkey, persistent-memory wiring, and capability-gated tool execution.

## 03 — The Interop Librarian (self-growing cross-system translation)

Researched + built 2026-10-11 against the actual stack, from the user's
"building as it goes translation library" directive. Four modules
(wubu_abi_path, wubu_syscall_shim, wubu_librarian, wubu_librarian_serve)
turn foreign OS apps into findable, callable participants with a
translation dictionary that learns itself from real crossings. Sits on
Plan 9/Styx (resources-as-files, user-level translators) + capability
security via namespace. Full notes in `03-interop-librarian.md`.

## 01 — Manifold geometry for neural substrates

### 1.1 Hyperbolic models: Poincaré ball vs hyperboloid vs Klein
**Sources:**
- Nickel & Kiela, *Poincaré Embeddings for Learning Hierarchical
  Representations*, NeurIPS 2017 — arXiv:1705.08039
- Chien et al., *Adaptive Universal Generalized PageRank Graph Neural
  Network*, ICLR 2022 — arXiv:2006.07988
- Mao et al., *Klein Model for Hyperbolic Neural Networks*, NeurIPS 2024
  Symmetry & Geometry workshop — arXiv:2410.16813
- Skopek, Ganea, Bécigneul, *Mixed-curvature Variational Autoencoders*,
  ICLR 2020 — arXiv:1911.08411

**What they do.** Three coordinate charts of the same constant-negative-
curvature space: Poincaré ball (angles conformal, metrics cheap),
hyperboloid (linear maps are clean, Minkowski inner product), Klein
(straight geodesics, cheap Einstein midpoint/midpoint-free ops).

**The transferable technique.** Klein-model scalar multiply and parallel
transport reduce to plain tangent-space ops (Einstein addition /
scalar multiplication) with NO exp/log map needed. Möbius ops in
Poincaré coordinates need exp0/log0 per op. For the nested-sphere WuBu
model this matters: an op on a nested sphere is an op on a point plus
N-1 radial components; Klein's straight geodesics make the radial part
a linear interpolation instead of an exp/log pair.

**Our delta.** WuBu's nesting is *concentric spheres*, not product
manifolds. A product of balls (Skopek) multiplies distances across
factors; nesting composes radii. The Poincaré ball remains the base
manifold per the user's doctrine ("the Poincaré sphere is a basic
manifold math network base"). Klein is a candidate chart for the
radial/nesting dimension specifically, not a replacement.

**Status:** overlap-strong (chart choice for the radial dimension).

### 1.2 Mixed-curvature product manifolds
**Sources:**
- Gu et al., *Learning Mixed-Curvature Representations in Product
  Manifolds*, ICML 2019 — arXiv:1806.04360
- Skopek et al., ICLR 2020 — arXiv:1911.08411
- *Curve Your Attention: Mixed-Curvature Transformers for Graph
  Representation Learning*, arXiv:2309.04082
- *CAT: Curvature-Adaptive Transformers for Geometry-aware ...*
  arXiv:2510.01634

**What they do.** Embed as a product H^n × S^n × E^n, each factor with
learnable curvature; attention runs per-factor then recombines.
Curvature-adaptive variants (CAT) let each head pick its curvature.

**Transferable technique.** The "learnable curvature per head/factor"
pattern is exactly the amoeba's width/depth mutation surface. Our
amoeba model mutates width; theirs mutates curvature. Both are a
smooth one-parameter family controlling capacity per region.

**Our delta.** We nest spheres (radial depth) rather than multiplying
curvatures. Both generalize Euclidean attention; the mathematics
(exp/log maps in each factor) is the shared machinery.

**Status:** overlap-strong (learnable-curvature-per-head mutation).

### 1.3 Hyperbolic KG embeddings — rotation vs beta vs Klein
**Sources:**
- Chami et al., *RotatE: Knowledge Graph Embedding by Relational
  Rotation in Complex Space*, ICLR 2019 — arXiv:1902.10197
- Balazevic et al., *Multi-relational Poincaré Graph Embeddings*,
  NeurIPS 2019 — arXiv:1905.09766
- *Fully Hyperbolic Rotation for Knowledge Graph Embedding*,
  arXiv:2411.03622
- *Geometric Relational Embeddings: Progress and Prospects* (survey)

**What they do.** Relations as rotations (phase in complex space) or
Möbius transformations (in the ball). Rotation keeps entity and
relation in different spaces; Möbius keeps everything in the ball so
composition is exact.

**Our relevance.** The WuBu hive/amoeba model's "spheres talk to each
other via a physics router" is a relation-composition problem. Möbius
composition in-ball is the candidate primitive; rotation is the
candidate for directional/asymmetric relations.

**Status:** candidate.

---

## 02 — Formal mathematics as a verification oracle

### 2.1 DeepMind AlphaGeometry / AlphaProof, IMO silver 2024
**Sources:**
- deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level/
- codeforces.com/blog/entry/131911 (community summary of the system)
- Trinh et al., *Solving olympiad geometry without human
  demonstrations*, Nature 2024 (AlphaGeometry) — the DD+AR loop

**What they do.** AlphaGeometry: a symbolic engine (DD) proposes
auxiliary constructions; a neural model (AR) ranks them; a formal
checker (Lean/Geo) accepts or rejects. AlphaProof: pretrained LM +
RL against a Lean prover; an informal-to-formal translation model
fronts it.

**Transferable technique.** The *propose/rank/verify* loop with a
HARD formal verifier as the only accept path. This is exactly the
"gauntlet" discipline in wubunos: the compiler is the verifier, the
model is the proposer. The RLHF oracle pattern already in WuBu training
uses a soft judge; the upgrade is a hard verifier for anything that
can be formalized (arithmetic identities, type checks, the Lean proofs
already in `wubuwizard/MATH/lean/`).

### 2.2 OpenAI formal-math (2024-2025)
**Source:** openai.com/index/formal-math/ — *Solving (some) formal math
olympiad problems*.

**What they do.** Fine-tuned an LM to emit Lean 4 proofs; an iterative
search loop with the Lean compiler as the judge; RL on proof success.

**Our overlap.** `MATH/lean/wubu_proofs` already has Lean-verified
theorems for the nesting theory. The gap: the training corpus has no
proof-search tasks. Adding "given goal, produce Lean proof, reward =
compiles" tasks to the SFT corpus is the direct transfer.

**Status:** overlap-strong for the corpus/oracle design.

### 2.3 OpenAI math repo (github.com/openai/math)
**Source:** github.com/openai/math (user-linked).

**What it is.** A benchmark harness for formal math: Lean/Metamath
dataloaders, proof-step extraction, evaluation of models on proof
success at given depth budgets. NOT a math-AGI system — a MEASUREMENT
tool.

**Our overlap.** Same category as the wubunos gauntlet_runner: a
reproducible, file-driven harness with pass/fail oracles. We can steal
their *interface design* (goal → tactic sequence → verifier) for a
Lean-gauntlet in wubunos without importing their implementation.

**Status:** overlap-strong (harness design), divergent (implementation
is C11 in wubunos, not Python).

---

## 03 — Systems-level comparisons (for the OS/compiler work)

### 3.1 ReactOS NT syscall surface — ordinal-driven transliteration
**Source of record (in-tree):** `wubuos/src/runtime/vsl/vsl_nt_*.c`
(21 translation units, 67 modules incl. headers) with ordinals taken
from `reactos-study/reactos/ntoskrnl/sysfuncs.lst` (line = 1-based),
per `vsl_nt_misc.c:5`. Module headers document Windows 11 24H2
ordinal ranges where they diverge from the ReactOS table
(`vsl_nt_alpc.c:411` — "These are Windows 11 (24H2) ordinals, not in
ReactOS sysfuncs.lst").
**What they do.** ReactOS reimplements the NT kernel as a free
Windows-compatible kernel: one syscall dispatcher (`NtSystemCall`),
a service table (`sysfuncs.lst` — ordinal → handler), and each
subsystem (objects, I/O, ALPC, worker) as a library behind that
contract. The ABI is the product.
**What we do.** The same architecture, hosted on Linux: a VSL
(Virtual System Layer) NT-bridge whose dispatcher resolves by NT
ordinal, with ~193 syscalls across 21 modules (process, thread,
token, job, registry, section, vmem, sync, timer, alpc, atoms,
wnf, ioring, enclave, worker, ktm, misc...), each a real
Linux/VSL implementation behind the NT contract. Cap-backed
authority (`vsl_nt_cap.c`) replaces the NT object manager for
handle→object resolution.
**Transferable technique.** The ordinal-table-as-ABI pattern: pin
the *dispatch contract* (number → behavior) and implement behind
it freely. That is why our 24H2-only ordinals can live beside the
ReactOS table without conflict — each module documents its source.
**Our delta.** We transliterate to Linux syscalls instead of
reimplementing a kernel; the bridge is a hosted runtime, not a
bare-metal OS. Authority is capability-based (wubu_cap) rather
than token/ACL-based, a deliberate security upgrade.
**Status:** overlap-strong (dispatch-contract design).

### 3.2 ZealOS/TempleOS — identity-mapped kernel, 1:1 name parity
**Sources:**
- zealos_parity.h (in-tree, wubuos/src/kernel/) — 96/96 core
  functions mapped 100%
- The header itself states the design rules: snake_case is
  primary, ZealOS PascalCase is a compile-time #define/typedef
  alias, zero cost, no conflicts
**What they do.** TempleOS/ZealOS: single-address-space,
identity-mapped kernel; HolyC compiled JIT into the same ring-0
space as the kernel; one language, one ABI, no protection
boundaries by design. "The compiler IS the OS."
**What we do.** Same philosophy, hosted and hardened: the HolyD
("My Seed") compiler in wubunos is a HolyC-family frontend whose
C11 backend shares our JIT; `zealos_parity.h` gives 1:1 name
mapping so ZealOS source reads alongside our C port with zero
mental translation.
**Transferable technique.** Name-parity tables as a *reading and
porting* tool: when porting a foreign codebase, a compile-time
alias header beats renaming — the original literature stays
citable line-by-line, and the primary implementation evolves
independently. Also the single-language-kernel idea survives as
the HolyD→C11→JIT path in our tree.
**Our delta.** We keep the ring-0/ring-3 split and cgroups v2 +
seccomp-bpf container isolation (`wubuos/src/runtime/container/`,
ct_iso_* + wubucontainer) — TempleOS's single-address-space is
deliberately not copied; the parity is at the API/naming layer
only.
**Status:** overlap-partial (philosophy shared, isolation model
divergent).

### 3.3 Proton/SteamOS — the container/exec runtime posture
**Source of record (in-tree):** `wubuos/src/runtime/container/`
(wubucontainer, ct_iso_cgroup/ns/seccomp, oci/), plus the
user's standing RUN POLICY (memory): third-party software runs ON
the AGI kernel via exec backends (wine/proton, arch/debian
containers in-wubuos); booting host Arch + host wine is cheating.
**What they do.** SteamOS/Proton: a Linux distro whose entire job
is running foreign Windows binaries in-game: a compat layer
(wine/proton) translating Win32→Linux, plus containerization
(Steam Linux Runtime) pinning the userspace so games see a
stable ABI across host updates. The pattern: *pin the userspace,
translate the ABI, run untrusted code sandboxed*.
**What we do.** The same three-part pattern: VSL NT-bridge
translates the Win32/NT ABI (3.1), wubucontainer + cgroups v2 +
seccomp-bpf pins and sandboxes the userspace, and the exec
backends run the payload inside the WuBuOS kernel — never the
host. The run policy above is the doctrine form of Proton's
"game runs on SteamOS, not on your distro".
**Transferable technique.** (a) Pinned runtime containers keyed to
the payload, not the host — our oci/ + wubucontainer_registry is
the same idea in C11. (b) ABI translation as a *library layer*
under the sandbox, so the translation surface can evolve
independently of the isolation surface. (c) The anti-cheat
posture: our "runs ON the AGI kernel" rule is the exact analogue
of Proton's anti-cheat requirement that the game runs under the
Steam runtime.
**Our delta.** Proton targets DX/ Win32 games; we target the NT
surface broadly (3.1) plus general Linux containers. The sandbox
mechanism is seccomp-bpf + cgroups v2 (kernel-enforced), stronger
than Proton's userspace-heavy runtime.
**Status:** overlap-strong (runtime posture + sandbox pattern).
