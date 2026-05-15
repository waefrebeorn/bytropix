# Diagrams (May 15 PM v6 — FINAL)

Architecture and pipeline diagrams for the WuBu Nesting project, created as SVG files.

## Available Diagrams

| Diagram | Description | Status |
|---------|-------------|--------|
| `phase-roadmap.svg` | Full project phase roadmap with timeline and metrics | **v6 — 100% complete** |
| `training-pipeline.svg` | Training pipeline: GGUF → Dequant → GPU → MoE → Proj → Loss → Flags | **NEW — 11s/step flow** |
| `tailslayer-pattern.svg` | Tailslayer hedged-read pattern mapped to speculative decoding | **NEW — spec-decode analogy** |
| `paper-audit.svg` | 14 Qwen3.6 architecture params vs C implementation cross-ref | **NEW — 9✅ 3🔍 2❌ 1⚠️** |
| `gguf-rip-pipeline.svg` | GGUF → C/CUDA pipeline with 7 math extensions | **v6 — 11s/step** |
| `wubu-math-pipeline.svg` | Euclidean → Poincaré → Möbius → Nested geometric pipeline | **v6 — P0-P2 all done** |
| `research-timeline.svg` | Complete research arc Aug 2025 → May 2026 (6 phases) | **v6 — updated** |
| `llamacpp-clone-infrastructure.svg` | How we fork, study, and extract from llama.cpp | **v6 — training 11s/step** |
| `hamilton-encoder-pipeline.svg` | Hamilton encoder: RGB → quaternion → BSP tree | Stable (architectural) |
| `wubu-nesting-architecture.svg` | Original 4-level nested hyperbolic sphere architecture | Stable (conceptual) |

## How to View
All diagrams are standalone SVG files. Open in any browser or SVG viewer.

## How to Modify
Open the SVG in any text editor, adjust coordinates/colors/text, save and reload in browser.

For mermaid diagrams (used in markdown), see:
- `THEORY/03-wubu-nesting-paper.md` — full architecture mermaid diagram
