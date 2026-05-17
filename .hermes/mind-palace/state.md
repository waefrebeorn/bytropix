# state — May 17 v11 — RoPE FIXED (root cause of anti-correlation)

## Status
- **RoPE formula WRONG** — FIXED ✅
  - Was using `-2*i/ROTARY_DIM` (i/32) instead of `-2*i/sec_dim` (i/11 or i/10)
  - cos-sim went from -0.456 to -0.016 (anti-correlation eliminated)
- **GGUF dims[0] is innermost** (GGML convention) — confirmed
  - `k + j*D_MODEL` access pattern is CORRECT for all weights
  - MoE experts ARE contiguous per expert (expert = slowest dim)
  - My earlier "interleaved stride" theory was WRONG — reverted
- **Remaining cos-sim -0.016** — likely accumulated FP differences across 40 layers
  - Each layer's SSM/GQA output has tiny numerical drift vs llama.cpp
  - Over 40 layers, these compound to produce a different trajectory
  - MoE output rms≈0.016 (small correction, not the main signal)

## What was ACTUALLY wrong
1. RoPE frequency formula: `powf(THETA, -2*i/ROTARY_DIM)` → `powf(THETA, -2*i/sec_dim)`
   - Section dimensions: 22, 22, 20 (not 64)
   - This was the root cause of the -0.46 cos-sim (anti-correlation)

## What I investigated and found CORRECT
- Output projection: `j * D_MODEL + k` ✓
- All weight accesses: `k + j*D_MODEL` ✓ (dims[0] innermost in GGML)
- MoE expert dequant: contiguous per expert ✓ (reverted back from stride)
- RMSNorm: verified vs numpy ✓
- Token embedding: reads correctly ✓
- Type enums: match llama.cpp ✓

## Next Steps
- Layer-by-layer comparison needed to find remaining numerical drift
- Compare SSM forward intermediate values vs llama.cpp reference
- Compare GQA attention scores vs reference
- FP precision: our code uses `float` throughout, llama.cpp uses `double` for accumulators
