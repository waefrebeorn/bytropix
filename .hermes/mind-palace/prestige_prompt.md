# Prestige Prompt — May 19, 2026 (04:15) — Phase 8.3-8.4

## Project: bytropix — Qwen3.6-35B-A3B-UD-IQ2_M
Phase 8.3 (Expert Prefetch) and 8.4 (Output Proj Split) complete.

## Phase 8.3: Expert Prefetch
- **Before:** 256 bytes/expert to L1 (0.03% of weight data)
- **After:** Full-stride ~264KB(gate/up) + ~392KB(down) = ~920KB/expert → L3 via _MM_HINT_T2
- 8 experts = ~7.4MB prefetched (fits L3 on modern CPUs)
- ~25K prefetches issued during attn compute window (~1ms)

## Phase 8.4: Output Proj Split
- **Before:** Sequential for N tokens (prefill)
- **After:** `#pragma omp parallel for if(N > 1)` on outer token loop
- Nested OMP disabled by default → inner quantized_matmul uses 1 thread per token
- Decode path (N=1) unaffected

## DA Finding: Cos-sim divergence tracking
Layer-by-layer comparison (ref vs our):
- L00-SSM: 0.8598 — divergent from first SSM layer
- L01-SSM: 0.7464 — worsens immediately
- L06-L31: 0.97→0.88 gradual decay (amplified quant noise)
- L32-L39: 0.88→0.46 sharp drop (systematic divergence)

Next: Add SSM intermediate dumps to llama.cpp for step-by-step comparison.
Hypothesis: quantized_matmul path differs from llama.cpp's ggml_mul_mat at sufficient precision to compound through SSM recurrence.

## Next Priority: Phase 9 — KV Cache for GQA
GQA layers (10 of 40) recompute full softmax attention from scratch each decode. For 256K context, this becomes O(n²) and prohibitive.

## Vault Papers Cross-Referenced
- Qwen3.6-35B_Arch_Reference.md — architecture confirmed
- unsloth-qwen3.6-quant-formula.md — per-tensor type breakdown
- QWEN3NEXT_TENSOR_LAYOUT.md — complete tensor layout verified
- research_old_20260518.md — attn_gate NOT present in GQA layers (verified by list_tensors)
