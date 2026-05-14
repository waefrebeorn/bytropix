# THEORY/ — The Research Lab

> **You are now entering the research lab, not a textbook.**
>
> The code in the rest of this repo is the *what*. These documents are the *why*.

Welcome to the intellectual engine room of the bytropix project. What you'll find in this folder is not polished, pre-digested theory meant for a journal's strict page limit. It's the raw thinking — the exploratory mathematics, the cross-domain analogies, the philosophical convictions — that drives every line of code in the `ENCODERS/`, `DRAFT/`, and `CUDA/` directories.

This is a living document set. Some files are full papers, some are sprawling theoretical brain-dumps, and some are working notes that trace the evolution of an idea in real time. Read them in the spirit they're offered: as an invitation to think alongside the research, not as a final pronouncement.

---

## What's Here — A Map of the Lab

### `01-foundational-philosophy.md`
**"The Geometry IS the Architecture"**

This is your starting point. It lays out the core WuBu (層疊嵌套) philosophy in the most accessible terms: why hyperbolic space is a natural home for hierarchical data, why rotations (`SO(n)`) in tangent spaces matter for modeling dynamics, and why nesting multiple adaptive geometries yields a more powerful inductive bias than any single flat or curved space could. It also introduces the `HAKMEMQController` — a Q-learning agent that tunes the model's own hyperparameters in real time, acting as "adaptive strain engineering" on the geometry itself.

Think of this as the manifesto. Read it first.

### `02-axiomatic-emergent-theory.md`
**The Axiomatic-Emergent Theory of Physical Law (Qpaper.MD)**

This is a deeper, more speculative piece by WaefreBeorn (Wubu) that steps *outside* the AI domain to ask: can we reverse-engineer the universe's source code? The Axiomatic-Emergent Theory (AET) proposes that the ~20 arbitrary constants of the Standard Model are not fundamental inputs but *derived outputs* from a smaller set of dimensionless axioms. It introduces the Wubu Formalism — a calculus for reducing physical quantities to canonical form — and applies it to black hole thermodynamics (the "Rosetta Stone" equation), the speed of light (reframed as a vacuum-engineering problem with FTL implications), dark matter (recast as a gravitational incompleteness), and cosmological decoherence.

This document is the deep-well philosophical companion to the WuBu Nesting work. It shares the same DNA: a belief that structure — geometric, axiomatic, emergent — is more fundamental than brute-force fitting. You'll see cross-pollination between the phi-based parameterizations in the WuBu code and the Golden Dynamic Axiom discussed here.

### `03-wubu-nesting-paper.md`
**WuBuHypCD: Nested Hyperbolic Spaces with Tangent Rotations**

This is the core technical paper. It formally introduces **WuBu Nesting (層疊嵌套)**, the recursively nested architecture of adaptive hyperbolic spaces (`H^{n_i}_{c_i,s_i}`) where dimensionality, curvature, and scale are learned. Key architectural contributions detailed here:

- **Boundary Sub-Manifolds** — learnable point sets that represent substructures within each hyperbolic level
- **Tangent Space Transitions** — inter-level movement happens in the flat Euclidean tangent spaces, not the curved manifolds themselves
- **Explicit `SO(n)` Rotations** — the inter-level transformation is decomposed into a learned rotation (implemented via quaternions for 4D, general `SO(n)` matrices otherwise) applied simultaneously to the data representation, boundary manifolds, and level descriptor
- **Relative Vectors, Level Descriptors, Spread Parameters, Intra-Level Tangent Flows** — a rich suite of mechanisms for carrying information between scales
- Connection to the **HypCD** (Hyperbolic Category Discovery) paradigm

This paper is the mathematical backbone of everything in the `ENCODERS/` directory. The `wubu_nesting_impl.py` and `wubu_nesting_example.py` files are direct code realizations of the ideas in this paper.

### `04-spatio-temporal-findings.md`
**WuBuNestingFindings5.19.25: Analogies from Hyperbolic Surfaces and Material Physics**

This is where the theory gets *inspired* — cross-pollinating with geometric topology and condensed matter physics. Key analogies explored:

- **The Separating Systole (`~2 log(g)` scaling)**: From random hyperbolic surfaces, the insight that characteristic geometric lengths scale logarithmically with complexity. This suggests principled scaling laws for WuBu's own curvature, scale, and learning rate parameters — a sub-linear scaling alternative to the phi-based parameterizations already present in the code.
- **Optical Hyperbolicity & Band Nesting**: From 2D transition metal ditellurides (TMDs), the insight that anisotropic optical transitions + electronic band nesting produce "computational hyperbolicity" — directional, resonant processing. This maps directly onto the idea that WuBu levels can learn to act as highly tuned, direction-selective filters.
- **Adaptive Strain Engineering**: The idea that the `HAKMEMQController` and other meta-control systems act as a form of strain engineering on the geometric fabric of the model, dynamically tuning it for optimal performance.

This paper bridges pure mathematical analogy and practical design intuition.

### `WuBu_Nesting.pdf`
The full PDF of the WuBu Nesting paper. A typeset, polished version of the ideas in `03-wubu-nesting-paper.md` and then some. If you prefer reading equations in proper LaTeX rendering, start here.

### `WuBuHypCD.tex`
The LaTeX source for the WuBuHypCD paper. For those who want to hack on the paper itself, rebuild it, or extract LaTeX snippets for their own writing.

### `references.bib`
The bibliography for the WuBu Nesting project. Includes citations across hyperbolic geometry (Nickel & Kiela, Ganea et al.), quaternion neural networks, HypCD, geometric topology (Parlier-Wu-Xue on the separating systole), condensed matter physics (Wang-Low on optical hyperbolicity in TMDs), and more. Useful if you're writing derivative work or want to chase citation trails.

### `papers/GAAD-WuBu-ST1.md` and `papers/GAAD-WuBu-ST2.md`
**GAAD: Golden Aspect Adaptive Decomposition** — a φ-inspired front-end for aspect-ratio agnostic frame decomposition. Stage 1 introduces Recursive Golden Subdivision and Phi-Spiral Sectoring. Stage 2 integrates GAAD with WuBu-ST for the full video framework.

### `papers/DFT-WuBu.md` and `papers/DCT-WuBu.md`
**Spectral-domain WuBu variants.** DFT-WuBu integrates discrete Fourier transforms with hyperbolic geometry for frequency-domain reasoning. DCT-WuBu uses discrete cosine transforms for compression-oriented representations.

### `math_viz/` — Runnable Math Proofs (7 scripts)
Python scripts that generate the key diagrams and numeric proofs from first principles. Each script is self-contained and verifiable:

| Script | Proves | Run Command |
|--------|--------|-------------|
| `01_nested_hyperbolic_spaces.py` | Nested Poincaré disks, φ-scaled curvatures | `python3 THEORY/math_viz/01_nested_hyperbolic_spaces.py` |
| `02_golden_ratio_decomposition.py` | GAAD: φ-subdivision + spiral sectors | `python3 THEORY/math_viz/02_golden_ratio_decomposition.py` |
| `03_poincare_clock.py` | Soul/Echo gradient recovery (error < 1e-15) | `python3 THEORY/math_viz/03_poincare_clock.py` |
| `04_lie_group_nesting.py` | WuBu as G-bundle, SO(n) connection | `python3 THEORY/math_viz/04_lie_group_nesting.py` |
| `05_fiber_bundle_proof.py` | Fiber bundle structure proof | `python3 THEORY/math_viz/05_fiber_bundle_proof.py` |
| `06_symplectic_optimizer.py` | Symplectic RSGD on T(H^n) | `python3 THEORY/math_viz/06_symplectic_optimizer.py` |
| `07_lean_certificate.py` | Lean formal certificate stub | `python3 THEORY/math_viz/07_lean_certificate.py` |

Requires `pip install matplotlib numpy`. All originate from `~/HASHMIND/bytropix/math_viz/`.

### `math_viz/lean/`
Lean formal proof files corresponding to the mathematical claims in the math_viz scripts. Currently a stub directory for future expansion.

### `math_viz/run_all.py`
Batch runner for all math_viz scripts.

### `WuBu Spatio-Temporal Nesting.md`
**The spatio-temporal extension.** This document extends the WuBu Nesting framework into the time domain with a dual-nested architecture:

- **Spatial WuBu (WuBu-S)**: A projective cascade of nested hyperbolic spaces that processes each frame into a compact, geometrically-informed feature vector `s_t`
- **Temporal WuBu (WuBu-T)**: A second nested stack that models the dynamics *between* these frame feature vectors over time, with its own rotations (`R_τ`), boundaries, and flows

This is the theoretical foundation for the video/diffusion models in `DRAFT/` — particularly `WuBuNestDiffusion_v0.10.1_OpticalFlow.py` and the `WuBuGAADHybridGen` family.

---

## Suggested Reading Order

If you're new to the project, here's the path that'll make the most sense:

```
01-foundational-philosophy.md
    ↓  (15 min — get the big picture)
02-axiomatic-emergent-theory.md
    ↓  (30 min — understand the deeper philosophical engine)
03-wubu-nesting-paper.md  (or WuBu_Nesting.pdf)
    ↓  (1-2 hrs — the core technical framework)
04-spatio-temporal-findings.md
    ↓  (30 min — the analogies that inspire the design)
"WuBu Spatio-Temporal Nesting.md"
    ↓  (45 min — the video/temporal extension)
```

**Short on time?** Read `01-foundational-philosophy.md` and then jump to the PDF (`WuBu_Nesting.pdf`). Come back to the others as you need them.

---

## How This Feeds the Rest of the Repo

The theory in this folder is not academic wallpaper. It is actively driving:

- **`ENCODERS/`**: The `hash-mind/` and `phase1-symmetric-encoder/` directories contain implementations of hyperbolic embeddings, nested rotations, and the geometric attention mechanisms described in the papers. The `wubu_nesting_impl.py` and `wubu_nesting_example.py` files are direct ports of the paper's mathematics into PyTorch/JAX.

- **`DRAFT/`**: The diffusion models (`WuBuNestDiffusion_v*`) and VAE-GAN models (`WuBuGAADHybridGen_v*`) are full-scale experiments that marry the WuBu Nesting geometry with generative architectures. The training heuristics, Q-controller logic, and spectral transform integrations all trace back to ideas in these theory documents.

- **Future CUDA Integration**: The geometric operations described here — hyperbolic exponential/log maps, quaternion rotations in tangent space, nested parallel transport — are prime candidates for custom CUDA kernels. Work in `CUDA/` will accelerate the bottleneck operations identified in the papers.

- **Demos and Visualizations**: The classroom visualizer at [wubu-sphere-visual.replit.app](https://wubu-sphere-visual.replit.app/) and the Poincaré disk visualizations in `DRAFT/wubu_results/` are direct visual realizations of the nested hyperbolic spaces described in `03-wubu-nesting-paper.md`.

---

## A Note on Generosity of Reading

Some of these documents are *dense*. Some use notation that shifts between drafts. Some make leaps between pure math, physics theory, and ML architecture that can feel jarring. This is intentional — the research is happening in the gaps between disciplines, and the notation is being invented as we go.

If something doesn't click, try the following:

1. **Read diagonally** — get the gist before the details
2. **Follow a thread** — if the separating systole analogy in `04` catches your interest, follow it to the Parlier-Wu-Xue paper via `references.bib`
3. **Cross-reference with code** — when a paper says "we apply a learned rotation `R_i` in the tangent space," find `wubu_nesting_impl.py` and see exactly how that's implemented

The goal isn't to understand every equation on first pass. It's to absorb the *geometric sensibility* that permeates the project — and then let that sensibility guide your own experiments.

---

*"We don't fight the geometry of the data. We build the architecture inside the correct geometry from the start."*

— WaefreBeorn, The WuBu Philosophy
