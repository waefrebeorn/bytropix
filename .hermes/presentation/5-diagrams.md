# Architecture & Research Diagrams Index

> Index of 7 SVGs at `../../DIAGRAMS/` — phase roadmap, data pipelines, architecture diagrams, and research timeline for the WuBuText AI project.

---

## 1. WuBuText AI — Phase Roadmap

**File:** `../../DIAGRAMS/phase-roadmap.svg`

Six-phase project roadmap (May 2026) showing the WuBuText AI buildout from embedding graft through CUDA optimization. Phases 1 (Embedding Graft) and 2 (Attention Port — CPU) are done ✓; Phase 3 (Training Loop) is active; Phases 4–6 planned. ⚠️ DA Audit May 13: GPU forward path produces zeros — old "9.53 tok/s" claim was false positive. CPU forward verified at CE 12.66. GPU weight loading bug (bench.c) is P0 fix. See devils_advocate_v3.md for full audit.

---

## 2. GGUF Representation Pipeline

**File:** `../../DIAGRAMS/gguf-rip-pipeline.svg`

End-to-end pipeline extracting Qwen3.6-35B-A3B weights from llama.cpp GGUF format into WuBu hyperbolic embeddings. Tracks the flow from GGUF tensor parsing → weight extraction (SSM/GQA layers) + embedding dequantization → Poincaré ball mapping (R=0.956) → quality verification (95% NN preservation). Culminates in the 40-layer model assembly — CPU forward verified (CE 12.66), GPU forward ⛔ blocked by weight loading bug (DA Audit May 13).

---

## 3. llama.cpp Clone Infrastructure

**File:** `../../DIAGRAMS/llamacpp-clone-infrastructure.svg`

Infrastructure diagram of how the WuBu team forks, studies, extracts from, and benchmarks against llama.cpp. Shows three workstreams: (A) Architecture Study of `qwen3next.cpp` for SSM recurrence formulas and tensor layouts, (B) GGUF Model Extraction for embeddings and weights, and (C) Benchmark Runner using llama-server for baseline comparison (34.9 tok/s baseline, 38.67 tok/s with Hamilton encoder). Lists everything taken from llama.cpp (GGUF format, CUDA patterns, tensor layouts) vs. what was built from scratch (hyperbolic geometry, TST training, Poincaré mapping).

---

## 4. WuBu Math — Geometric Pipeline

**File:** `../../DIAGRAMS/wubu-math-pipeline.svg`

Mathematical pipeline tracing the Euclidean-to-hyperbolic transformation: Qwen token embeddings (248320×2048) → Poincaré ball map (exp_map, R=0.956) → SSM Gated Delta Net (30/40 layers) with recurrence formulas → GQA Full Attention (10/40 layers) → Poincaré Hyperbolic Recurrence → Möbius Gyration in tangent space (SO(4) rotation) → future Nested Hyperbolic Hierarchy with per-level curvature, scale, and rotation.

---

## 5. WuBu Nesting (層疊嵌套) Architecture

**File:** `../../DIAGRAMS/wubu-nesting-architecture.svg`

Architecture of the nested hyperbolic spaces system with four hierarchical levels: Level 1 (Coarse Structure / Global Context), Level 2 (Parts & Components / Meso Features), Level 3 (Substructures / Fine Details), and Level 4 (Atomic / Core Features). Key components listed include Adaptive Curvature cₙ, Scale sₙ, Boundary Manifolds Bᵢⱼ, Rotation Rₙ ∈ SO(n), Descriptors ldₙ, Spread σₙ, Flow Fₙ, and Relative Vectors dₙⱼₖ.

---

## 6. The Hamilton Encoder Pipeline

**File:** `../../DIAGRAMS/hamilton-encoder-pipeline.svg`

Vision encoder pipeline from RGB pixels to quaternion-encoded latent vectors on CUDA. Steps: Input RGB tensor → 2×2 Avg Pool (RGB→HSL conversion) → Quaternion Encode (H→w, S→x, L→y) → Quaternion(5-channel) output. Integrated with BSP Tree quaternion-split indexing (O(log N) KV retrieval) and GPU KV cache fused attention. Performance metrics on RTX 5050: 39.9 t/s generation, 137.2 t/s prefill, ~62% KV cache memory reduction, ~3% encoder overhead.

---

## 7. Research Timeline: The WuBu Nesting Journey

**File:** `../../DIAGRAMS/research-timeline.svg`

Chronological research journey from August 2025 to April 2026. Five milestones: (1) Aug 2025 — Axiomatic-Emergent Theory physics paper with κ-factor and FTL vacuum engineering; (2) Sep 2025 — Neural encoders including topological autoencoders and Chimera ResNet; (3) Sep-Oct 2025 — Multi-modal expansion with audio synthesis, video diffusion, CLIP-guided generation; (4) Nov 2025 — Geodesic AI Brain with 20+ geodesic variants; (5) Jan-Apr 2026 — WuBu Nesting paper → CUDA kernels yielding 39.9 t/s at 256K context. Includes commit highlights from the project diary.
