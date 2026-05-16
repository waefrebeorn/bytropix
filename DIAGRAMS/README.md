# Diagrams (May 17 PM v7 — Corrected)

Architecture and pipeline diagrams for the WuBu Nesting project, created as SVG files.

## Available Diagrams

| Diagram | Description | Status |
|---------|-------------|--------|
| `phase-roadmap.svg` | Full project phase roadmap (May 17 — corrected) | **v7 — all dequants verified** |
| `inference-pipeline.svg` | Current inference debug pipeline with quant types | **NEW — May 17** |
| `quant-layer-map.svg` | Down_exps tensor types by layer (DA v9 corrected) | **NEW — May 17** |
| `bug-status.svg` | Dequant verification status vs llama.cpp | **NEW — May 17** |
| `training-pipeline.svg` | Training pipeline: GGUF → Dequant → GPU → MoE → Proj → Loss → Flags | v6 |
| `tailslayer-pattern.svg` | Tailslayer hedged-read → speculative decoding | v6 |
| `paper-audit.svg` | 14 Qwen3.6 params vs C implementation cross-ref | v6 |
| `gguf-rip-pipeline.svg` | GGUF → C/CUDA pipeline with 7 math extensions | v6 |
| `wubu-math-pipeline.svg` | Euclidean → Poincaré → Möbius → Nested geometric pipeline | v6 |
| `research-timeline.svg` | Complete research arc Aug 2025 → May 2026 | v6 |
| `llamacpp-clone-infrastructure.svg` | How we fork/study/extract from llama.cpp | v6 |
| `hamilton-encoder-pipeline.svg` | Hamilton encoder: RGB → quaternion → BSP tree | Stable |
| `wubu-nesting-architecture.svg` | Original 4-level nested hyperbolic sphere architecture | Stable |

## Key Updates (May 17)
- **Revised** `phase-roadmap.svg` — reflects current inference debug focus, not training
- **New** `inference-pipeline.svg` — shows the per-layer flow with actual quant types
- **New** `quant-layer-map.svg` — per-layer quant type with DA v9 corrections
- **New** `bug-status.svg` — dequant verification table with SSM divergence note

> DA v9: Python `dump_gguf.py` had WRONG type labels. Now corrected. See `.hermes/mind-palace/plans/devils_advocate_v9.md`

## How to View
All diagrams are standalone SVG files. Open in any browser or SVG viewer.

## How to Modify
Open the SVG in any text editor, adjust coordinates/colors/text, save and reload in browser.
