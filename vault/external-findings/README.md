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

Entries to fill as gaps close. Sources of record:
- ReactOS (`vendor`-adjacent study trees, NT ordinal tables)
- ZealOS/TempleOS (identity-mapped kernel model)
- SteamOS/Proton (the container/exec runtime work)

*(No entries yet — reserved.)*
