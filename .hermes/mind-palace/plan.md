# Plan — Feature Cream Roadmap: Phase 28f

## ✅ Completed (This Session)
1. **GPU output projection FIXED** — Q4_K dequant layout was wrong (used V-stride instead of D-stride, stored as [D][V] but cuBLAS expected [V][D]). F32 SGEMM mode now correct.
2. **SSM state sync FIXED** — Added wubu_gpu_sync_ssm_state_to_gpu/cpu functions. CPU→GPU sync after hybrid prefill, GPU→CPU sync after forward_full decode. Also sync before hybrid decode.
3. **Hybrid decode path with state sync** — forward_full disabled, hybrid recurrence active with correct state.

## 🔴 P0: Fix forward_full GPU SSM Divergence
forward_full produces wrong output. Debug tests show:
- First call: near-zero output (conv_state all-zero → small conv → small recurrence → small output)
- With state accumulation: non-zero but wrong values
- Forward_full disabled; hybrid path used instead

**Debug approach:**
1. Compare GPU vs CPU conv output for first token (both start zero-state)
2. Compare GPU vs CPU recurrence for first token
3. Build step-by-step debug dump

## 🟡 P1: Infrastructure
1. **Fix CPU gen_text build** — wrap GPU symbols in `#ifdef GPU_SUPPORT` in wubu_model.c
2. **Push 8 commits** to `waefrebeorn/bytropix` (master) — current fixes are critical
3. **Re-enable forward_full** after debug

## 🟡 P2: Vision Verification
1. Build test_vision_real
2. Run vision E2E
3. Wire multi-modal pipeline

## Key Files Reference
| File | Purpose | Status |
|------|---------|--------|
| `src/wubu_model_gpu.cu` | GPU SSM forward + state sync | ✅ Fixed |
| `src/gpu_output_proj.cu` | GPU output projection (F32 SGEMM) | ✅ Fixed |
| `include/wubu_model.h` | Declarations | ✅ Updated |
| `src/wubu_model.c` | Dispatch logic + state sync | ✅ Fixed |
| `src/wubu_ssm.c` | CPU SSM forward | ✅ Verified vs llama |
| `src/gpu_ssm_recurrence.cu` | GPU recurrence kernel | ❌ Suspect |
