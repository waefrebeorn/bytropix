# DIAGRAMS — Architecture and Pipeline Visualizations

SVG diagrams for the bytropix inference engine, phase roadmap, and status.

## Current Diagrams

| Diagram | Description | Last Updated |
|---------|-------------|-------------|
| `inference-pipeline-v22.svg` | Full inference pipeline: per-layer data flow with quant types, cos-sim status badges, 3:1 SSM/GQA interleaved layers, Q4_0 KV cache, honest status indicators | **May 19 PM (v22)** |
| `phase-roadmap.svg` | Full project phase roadmap (May 17 — may be stale) | May 17 |
| `bug-status.svg` | Dequant verification status vs llama.cpp | May 17 |
| `quant-layer-map.svg` | Down_exps tensor types by layer | May 17 |
| `status-may19-2026.svg` | Status snapshot from May 19 | May 19 |
| `paper-audit.svg` | Qwen3.6 params vs C implementation cross-ref | May 17 |
| `research-timeline.svg` | Research arc Aug 2025 → May 2026 | May 17 |
| Other | Training pipeline, tailslayer, gguf-rip, math pipeline, hamilton encoder | May 15-17 |

## How to View
Open any SVG in a browser. All are standalone files.

## How to Create/Update
Write a new SVG and add to the table above. Use the inference-pipeline-v22.svg as a style reference (dark background #1a1a2e, bright colors, monospace font).
