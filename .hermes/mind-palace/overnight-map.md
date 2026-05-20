# Overnight Map — Phase 20: MoE Expert Cache on GPU

## State: Decode 8.9 tok/s, Prefill 18.6 tok/s on 8GB Laptop GPU

Phase 20 complete: **MoE expert cache on GPU**. Per-layer persistent cache stores last-used 8 experts' weight data (gate/up/down). On routing stability (expert indices unchanged for same layer between tokens), zero H2D transfers — the MoE kernel reads directly from GPU cache.

## What Was Done
- `moe_cache_eid[40][8]` — per-layer expert index tracking
- `moe_cache_w[40][3][8*270KB]` — per-layer GPU weight blobs
- Cache check in `wubu_model_gpu_moe_experts`: compares indices, skips H2D on hit
- `wubu_gpu_moe_forward_experts` accepts `use_gpu_ptrs` flag for direct GPU pointer pass-through
- On miss: H2D → scratch → kernel → D2D cache update
- 259MB cache fits within 8GB VRAM budget

## Remaining Targets
1. **Sparse/streaming attention** — O(n·k) for >256k GQA attention (last big bottleneck at 256k)
2. **Unified SSM forward kernel** — fuse all SSM into single kernel
