# Overnight Map — Phase 28c: GPU SSM Path Fixed, Now Verify Correctness

**Active repo**: /home/wubu/bytropix/
**Model**: Qwen3.6-35B-A3B-UD-IQ2_M.gguf (qwen35moe arch)
**Current state**: GPU SSM C==1 decode path RUNS without illegal access ✅. Output is garbage (`<-1>` tokens) 🔴.

## The Critical Bug That Was Found
All prior "SSM GPU path never worked" issues traced to ONE bug:
- Lines 417-422: `for (int i = 0; i < 40; i++) gpu->d_ssm_beta[i] = NULL` etc. INSIDE the per-layer loop
- Each new layer NULL'd the previous layer's allocations
- Only L38 (last SSM) had valid small F32 weights
- `ssm_beta_alpha_fused_decode` crashed on NULL GPU pointer → "illegal memory access"

## Changes This Session
1. `#if 0` wrapped F32 dequant upload (saves ~2.2 GB VRAM)
2. ssm_project column-major → row_major
3. **REMOVED** per-iteration NULL re-zero loops (THE BUG)
4. Added CUDA error checks + NULL checks throughout forward_full
5. Added debug MARK prints (will clean up)

## Verified Working
- All 30 SSM layers through forward_full C==1: beta/alpha kernel ✅, conv/silu/split ✅, L2 norm ✅, recurrence ✅, SiLU/gated norm ✅, ssm_out matmul ✅
- No illegal memory access, no CUDA errors

## What Produces Garbage
- Output is `<-1><-1><-1>` — token ID -1 from top-k with NaN/-inf logits
- Possible causes:
  1. SSM GPU compute path produces wrong hidden states (numerical error)
  2. Output projection produces NaN/inf logits
  3. GQA attention also has GPU path that might be wrong
  4. Residual connections wrong (GPU modifies attn_out only, not x)

## Debug Next: Isolate to SSM vs GQA vs Output
1. Test with CPU-only gen_text (too slow — 120s+)
2. Use llama.cpp ref_dumper for reference data
3. Compare GPU vs CPU layer outputs via DUMP_LAYER_DIR
4. Check if the row_major quant matmul produces correct values (compare vs CPU quantized_matmul)
