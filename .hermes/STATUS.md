# WuBu Nesting (層疊嵌套) — Hermes Vault

## Entry
```
.hermes/index.md —→ vault/attention/ —→ vault/encoders/ —→ vault/hash-mind/
                  → vault/theory/      → vault/phase3/    → vault/hamilton/
                  → vault/diffusion/   → vault/audio/     → vault/draftPY/
                  → vault/c-training/  → vault/optimizers/→ vault/lean-proofs/
```

## Research Timeline (commit diary)
```
Aug 19  — gemini went hawking (physics theory starts)
Sep 06  — THIS WORKS AND SHOULD BE RESEARCHED
Sep 11  — XJDR SAMPLER IMAGE TOKENIZATION
Sep 21  — deepseek saw the paper and wrote a paper
Sep 30  — BOOM SHAKA LAKA AUDIO AND TEXT
Oct 04  — WE PUT IN WORK
Oct 29  — I GIVE THIS TO THE WORLD
Oct 31  — daddy came back with sci fi math
Nov 22  — VALIDATED "Energy-Based Manifold Learning"
Nov 26  — "geodesic ai brain" lol its a sphere bruh
Jan 28  — WuBu Nesting Paper drop commit
Mar 09  — I solved video and audio in one morning
Apr 04  — quantum work tracking
May 12  — massive reorg: structures into THEORY/MATH/ENCODERS/...
```

## Current Active Tracks (2026-05-12)
1. **C Training** — `ENCODERS/hash-mind/c/` — Pure C transformer training (210K params, ~4000 steps/s)
2. **Lean Proofs** — `MATH/lean/wubu_proofs/` — mathlib4 building, 4 proof files
3. **LLAMA-CPP Integration** — External repo at `~/HASHMIND/llama-cpp-rotorquant/llama.cppCOPY/`
   - MLP encoder LIVE: 39.9 tok/s gen, 137 t/s prefill
   - 20/20 milestones, 40-turn stress test passed
   - P6 NestedWuBu staged occlusion: O(N²) → O(√N)

## Topology of the Repo
```
bytropix/
├── THEORY/               Physics → philosophy → paper (4 docs, 1 PDF, 1 LaTeX)
├── MATH/                 Wubu formalism + Lean proofs
├── ENCODERS/             The research core (5 phases + hash-mind + hamilton)
│   ├── phase1-symmetric-encoder/   Geometric AE, .wubu format
│   ├── phase2-topological-ae/      Quantum autoencoder, 3-float compression
│   ├── phase3-generative/          VQ-VAE + Conductor transformer + CORPUS.py
│   ├── hash-mind/                  WuBuMind V1-V7.1 JAX, SimpleHash precursors
│   ├── hamilton-encoder-cpu/       Geodesic AI Brain (30+ Python files)
│   └── hash-mind/c/                C port: train.c + manual backprop
├── DIFFUSION/            HGA UNet + funnel diffusion
├── AUDIO/                WubuSynth galactic core synthesizer
├── ATTENTION/            4 variants: sparse, hyperbolic, topological, entropix
├── OPTIMIZERS/           Q-Controller, PID, toroidal gradient
├── DIAGRAMS/             SVG research timeline, architecture diagrams
├── DRAFT/                Batch files, setup scripts
├── draftPY/              40+ experimental scripts (GAAD, SpecTrans, HypCD)
└── .hermes/              This vault
```
