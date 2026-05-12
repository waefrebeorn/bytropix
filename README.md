# bytropix — WuBu Nesting (層疊嵌套)

**Adaptive, Rotation-Aware, Nested Hyperbolic Architectures for AI**

---

> This is a research lab notebook, not a polished library. Every file here represents a moment of discovery, a failed experiment, or a breakthrough. The value is in the ideas and the journey.

---

## What is this?

bytropix is the research archive of the **WuBu Nesting (層疊嵌套 — céngdié qiàntào)** project: a geometric deep learning framework that builds AI architectures *inside the correct geometry of the data* rather than forcing data through flat Euclidean layers.

Instead of billion-parameter brute force, WuBu uses:
- **Nested hyperbolic spaces** (Russian doll geometry) with learnable curvature and scale
- **Quaternion rotations** (SO(4)) in tangent spaces between levels
- **BSP trees** (binary space partitioning) for logarithmic-time attention
- **Hamilton encoders** that map data to quaternion representations

This repo contains the complete research arc — from a physics theory paper in August 2025 to working JAX/Python prototypes for AI representation compression.

---

## The Research Arc

```
Aug 2025 — Physics Theory (Axiomatic-Emergent Theory, Wubu Formalism)
Sep 2025 — Symmetric Neural Encoders (Phase 1-3, 512×512 autoencoders)
Sep 2025 — HashMind & Rolling Hash Attention
Oct 2025 — Audio/Video Generation, Diffusion Models
Nov 2025 — Geodesic AI Brain (20+ geodesic layer experiments)
Jan 2026 — WuBu Nesting Paper (full academic paper + LaTeX)
Mar 2026 — Audio & Video "Solved"  
Apr 2026 — BSP tree + quaternion encoding for AI compression
```

[View the full timeline in SVG](./DIAGRAMS/research-timeline.svg)

---

## Repository Map

```
bytropix/
├── THEORY/              ← The philosophy and academic papers
│   ├── 01-foundational-philosophy.md    "The Geometry IS the Architecture"
│   ├── 02-axiomatic-emergent-theory.md  Physics: Wubu Formalism, FTL, κ-factor
│   ├── 03-wubu-nesting-paper.md         The full WuBu Nesting paper
│   ├── 04-spatio-temporal-findings.md   From hyperbolic surfaces to AI
│   ├── WuBu_Nesting.pdf                The actual PDF
│   └── WuBuHypCD.tex                   LaTeX source
│
├── MATH/                ← The mathematical foundations
│   └── wubu-formalism.md              Q = Σ q_k ∏ α_i^E
│
├── ENCODERS/            ← The heart of the research
│   ├── phase1-symmetric-encoder/       Symmetric geometric autoencoders
│   ├── phase2-topological-ae/          Holomorphic Quantum Autoencoders
│   ├── phase3-generative/              Text-to-image generative pipelines
│   ├── hash-mind/                      Rolling hash attention, WuBuMind JAX
│   └── hamilton-encoder-cpu/           Geodesic layers, Chimera ResNet, CPU prototypes
│
├── DIFFUSION/           ← Hyperbolic geometric attention for generation
│   ├── hga-unet/                       Hyperbolic Geometric Attention UNet
│   └── funnel-diffusion/               Funnel diffusion, CLIP video
│
├── AUDIO/               ← Unsupervised adversarial audio synthesis
│   └── wubusynth/                      Galactic Core synthesizer
│
├── ATTENTION/           ← Beyond dot-product attention
│   ├── wubu-sparse-attention/          RAS indexer working/associative memory
│   ├── hyperbolic-attention/           Clockwork attention, kNN hyperbolic
│   ├── entropix-sampler/               Dynamic entropic sampling
│   └── topological-sequence-model/     Linear-complexity topological attention
│
├── OPTIMIZERS/          ← Meta-learning optimizers
│   └── q-controller/                   Q-learning LR/momentum control, PID
│
├── LLAMA-CPP-INTEGRATION/  ← CUDA GPU integration (reference)
│   ├── hamilton-encoder-cuda/          RGB→HSL→quaternion CUDA kernel
│   ├── bsp-tree-cuda/                  BSP tree quaternion-split on GPU
│   └── expert-cache/                   PCIe MoE expert cache
│
├── DIAGRAMS/            ← SVG educational diagrams
│   ├── wubu-nesting-architecture.svg
│   ├── hamilton-encoder-pipeline.svg
│   └── research-timeline.svg
│
├── MEDIA/               ← Generated images from the models
│
└── DRAFT/               ← Batch files, setup scripts, drafts
```

---

## How to Navigate This Repo

### Start Here (Newcomers)

1. **`THEORY/01-foundational-philosophy.md`** — 5-minute read. Understand the *why*.
2. **`DIAGRAMS/wubu-nesting-architecture.svg`** — Visual overview of the architecture.
3. **`ENCODERS/hash-mind/wubu_nesting_impl.py`** — The actual implementation with:
   - HyperbolicUtils (exp/log maps with scale awareness)
   - Hamilton product (quaternion rotation)
   - WuBuNestingLayer (the core layer)
4. **`LLAMA-CPP-INTEGRATION/README.md`** — How it all becomes CUDA.

### For Theory Readers

- **`THEORY/02-axiomatic-emergent-theory.md`** — Physics: the Wubu Formalism
- **`THEORY/03-wubu-nesting-paper.md`** — The full academic paper (515 lines)
- **`MATH/wubu-formalism.md`** — The central equation

### For Practitioners

- **`ENCODERS/hash-mind/WuBuMindJAX.py`** — Full JAX implementation with hyperbolic kNN attention
- **`ENCODERS/hamilton-encoder-cpu/chimera_quaternion.py`** — Quaternion attention
- **`ATTENTION/wubu-sparse-attention/WuBuSparseAttention.py`** — Working/associative memory
- **`OPTIMIZERS/q-controller/qcontroller.py`** — HAKMEM Q-learning optimizer

### For CUDA Engineers

- **`LLAMA-CPP-INTEGRATION/README.md`** — Full pipeline documentation
- **`DIAGRAMS/hamilton-encoder-pipeline.svg`** — Pipeline visualization

---

## The Commit Diary

The commit history of this repo is a raw, unfiltered research diary. Each message captures a moment:

| Date | Message |
|------|---------|
| 2025-08-19 | `gemini went hawking last night` |
| 2025-09-06 | `THIS WORKS AND SHOULD BE RESEARCHED` |
| 2025-09-11 | `XJDR SAMPLER IMAGE TOKENIZATION` |
| 2025-09-21 | `deepseek saw the paper and wrote a paper` |
| 2025-09-30 | `BOOM SHAKA LAKA AUDIO AND TEXT` |
| 2025-10-04 | `WE PUT IN WORK` |
| 2025-10-29 | `I GIVE THIS TO THE WORLD` |
| 2025-10-31 | `daddy came back with sci fi math` |
| 2025-11-22 | `VALIDATED "Energy-Based Manifold Learning or Neuromorphic Topology"` |
| 2025-11-26 | `"geodesic ai brain" lol its a sphere bruh` |
| 2026-01-28 | `WuBu Nesting Paper drop commit` |
| 2026-03-09 | `I solved video and audio in one morning` |
| 2026-04-04 | `quantum work tracking` |

This is not git hygiene — this is a research diary. Every commit tells a story of what was discovered, broken, fixed, or screamed at that day.

---

## The Core Philosophy

> Standard AI learns by brute force. It uses immense, billion-parameter models to approximate relationships in data, often inefficiently and without a true understanding of intrinsic structure.
>
> *This is like trying to flatten a globe onto a piece of paper — you will always have distortion and lose essential information.*
>
> The WuBu philosophy is different. We don't fight the geometry of the data; **we build the architecture inside the correct geometry from the start.**

— PHILOSOPHY.md

---

## Key Frameworks & Libraries Used

- **PyTorch** — Most encoder and attention experiments
- **JAX/Flax** — WuBuMind, diffusion, audio, geodesic layers
- **CUDA** — Hamilton encoder kernel, BSP tree, expert cache
- **llama.cpp** — Integration target for production inference
- **EnCodec (Meta)** — Audio tokenization backbone
- **CLIP** — Text conditioning for diffusion models

---

## Educational Value

This repo is valuable for anyone learning about:

- **Hyperbolic geometry for ML** — Working implementations of Poincaré ball exp/log maps
- **Geometric deep learning** — From theory to CUDA in one repo
- **Quaternion neural networks** — Hamilton product, SO(4) rotations
- **Attention mechanisms** — Sparse, hyperbolic, topological, entropic
- **Research methodology** — Raw, unfiltered research process
- **CUDA kernel development** — How theory becomes GPU code

---

## The Mission

| The WuBu math is the product — a compression framework for AI representations.
| No servers, no APIs, no CUDA benchmarks. Pure math. Pure geometry.
| 
| **Live API (public):** `https://insured-despite-editors-offering.trycloudflare.com`
| **Docs:** `/docs` endpoint
| **Payment:** BTC → `36sPuujTrcQN24G2NHDbcrARTtEYqyxzdP` or CashApp `$ManGamer`
| **Pricing:** $0.05/1K calls (embed), $0.10/1K (nest), $0.15/1K (GAAD)

---

## License

This is research code. Use it to learn, experiment, and build upon. Attribution appreciated but not required — science moves forward when ideas are shared.

---

> *"I GIVE THIS TO THE WORLD"* — Oct 29, 2025 commit message
