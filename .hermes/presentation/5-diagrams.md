# Architecture & Research Diagrams Index (May 15 PM v6)

> **10 SVGs** at `../../DIAGRAMS/` — 3 new, 7 updated to v6.

## 1. Phase Roadmap
Full project timeline: Phases 0-6 complete ✅. NaN fixed, 11s/step, 16× improvement.

## 2. Training Pipeline (NEW)
6-stage training flow: GGUF Load → Per-Expert Dequant → GPU Forward (30 SSM + 10 GQA) → MoE Router → Output Projection (cublasSgemm 248320×2048) → CE Loss + Backward. Metrics sidebar: 11s/step, CE 21.6→18.4, 0 NaN.

## 3. Tailslayer Pattern Match (NEW)
8-row analogy mapping: Tailslayer hedged-read concepts (N replicas, clflush+reload, first-response-wins) → speculative decoding analogs (N drafts, forward pass timing, longest-valid-prefix accept). Two-column dark theme with annotated arrows.

## 4. Paper Audit (NEW)
14 Qwen3.6 architecture parameters cross-referenced vs C headers. Color-coded: 9 ✅ Match (green), 3 🔍 Verify (amber), 2 ❌ Missing (pink), 1 ❌ Discrepancy (red). CONV_DIM 8192 vs 1536 highlighted.

## 5. GGUF → C/CUDA Pipeline
Updated v6: all 7 modules integrated, NaN fixed, vision 99ms 0 NaN.

## 6. llama.cpp Clone Infrastructure
Stale conceptually but still accurate. Metrics not refreshed.

## 7. WuBu Math — Geometric Pipeline
Euclidean→Poincaré→Möbius→Nested. v6: P0-P2 all done.

## 8. WuBu Nesting Architecture
Conceptual 4-level hierarchy — still valid as reference.

## 9. Hamilton Encoder Pipeline
Stable. Separate project artifact.

## 10. Research Timeline
Aug 2025 → May 2026 arc. Phase 6: "All 7 Cold Gaps Closed + NaN Fixed (May 14-15 2026)".
