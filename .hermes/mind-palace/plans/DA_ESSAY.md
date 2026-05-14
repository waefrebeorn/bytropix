# DA Audit Essay: The False Positive That Hid Half the Project

## Date: May 13, 2026 | Author: Devil's Advocate v3

---

## Prologue: How a Zero Passed as Verified

Every project has a blind spot. This one was `bench_e2e`.

The binary runs 40 layers sequentially on CPU, then GPU, then compares final hidden states and reports "PASS" if they match. It was built to verify the GPU forward against the CPU forward. For weeks, it had been producing this output:

```
CPU final[0:8]: +0.000000 +0.000000 +0.000000 +0.000000 ...
GPU final[0:8]: +0.000000 +0.000000 +0.000000 +0.000000 ...
Max diff (GPU vs CPU): 0.000000
PASS: GPU/CPU match within tolerance
Speedup: 29.69x
```

This looks like a PASS. Two paths producing identical output, a good speedup ratio. The problem: **both paths produced zero**. The comparison was matching on nothing — two null outputs differ by 0.000.

How did this happen? The CPU path in `bench_e2e` opens the GGUF file and reads weights **per layer, 40 times**. Each layer's weights are dequantized from GGML format. The GPU path does the same, copying to device memory each time. Both paths dequantize from the same GGUF — if the dequant is wrong, both produce garbage. And they do it identically, so they match at 0.000 diff.

The CPU forward in `train_real` uses a different path: `wubu_model_init` pre-loads all 733 tensors into a model struct, then `wubu_model_forward_from_embd` iterates layers using pre-loaded weights. This path **works correctly** — CE loss 12.66, logits non-zero. The difference: `wubu_model_init` was updated with the Q4_K fix; the per-layer loading in `bench_e2e` and `bench.c` was not tested independently.

## The Real Root Cause

The project spent weeks chasing "IQ2 dequant garbage" because NaN appeared in SSM forward, MoE output showed ±1e6 range, and CE loss was 6.6e10. All signs pointed to 2-bit quantization errors. The analysis was: "16 of 17 model tensors are IQ2_M — their dequant must be wrong."

The actual root cause: **Q4_K dequant had a wrong block size.** `block_q4_K` in modern llama.cpp is 144 bytes (no `qh` field). The GGUF reader used 176 bytes (with `qh`), which meant:
- Every Q4_K tensor (including `output.weight` [2048, 248320]) was read from wrong offsets
- Output projection produced values ±47M
- CE loss hit 6.6e10 (overflow)
- Gradients became NaN
- Everything downstream broke

Once fixed (commit fe8c17c): CE loss dropped from 6.6e10 → 87 → 12.7. MoE output went from ±1e6 → [-0.028, 0.031]. All NaN disappeared. The CPU forward was correct.

But `bench_e2e` and `gpu_load_ssm_layer` were never re-tested after this fix. They used the same `gguf_read_tensor_f32` function, so the dequant itself was correct. But the per-layer weight loading in `bench.c` reads tensors from a re-opened GGUF context, and this path has a separate bug — likely wrong tensor offset or dimension handling that survived the Q4_K fix.

## What the Audit Found

Running all 8 key binaries with fresh output:

| Binary | Before Audit (claimed) | After Audit (actual) |
|--------|----------------------|---------------------|
| train_real | "CE loss commented out" | CE 12.66 ✅ |
| test_moe | "IQ2 garbage output" | [-0.028, 0.031] ✅ |
| bench_e2e | "29x speedup PASS" | ALL ZEROS ⛔ |
| train_gpu | (not claimed) | CE 69 vs 12.66 ⛔ |
| train_backprop | (not claimed) | HANGS ⛔ |
| test_fused_vs_old | "PASS" | PASS diff 0.036 ✅ |
| test_tokenizer | "CJK works" | 你好→109266→你好 ✅ |
| dump_mmproj | "PASS" | 334 tensors ✅ |

**5 binaries work correctly. 3 have bugs. All 3 bugs trace to the GPU weight loading path or train_backprop hang.**

## Implications

### Stale Document Claims

The following documents had "verified" claims that were false positives:

1. **master_impl_plan_v2.md** — "Phase 2.5 GPU test: ✅ complete, 9.53 tok/s, 47.83x" → ⚠️ REVOKED
2. **presentation/4-implementation-status.md** — "GPU/CPU logit-level agreement confirmed" → ❌ FALSE
3. **presentation/1-project-overview.md** — "Phase 2.5 Complete" → ⚠️ Partial
4. **presentation/5-diagrams.md** — Two diagrams referenced "9.53 tok/s" → ⚠️ Corrected
5. **prestige_prompt.md** — "CE loss commented out", "IQ2 dequant garbage" → ❌ STALE
6. **overnight-map.md** — "IQ2 dequant the real blocker" → ❌ STALE, Q4_K was root

All have been patched or flagged with DA-correction notes.

### Energy Re-direction

Before this audit, the team believed:
- "GPU works at 9.53 tok/s" → focus on MoE integration
- "CE loss disabled" → focus on wiring loss
- "IQ2 garbage" → focus on IQ2 dequant

After audit:
- GPU forward is BROKEN — focus on bench.c weight loading (P0)
- train_real already has CE loss at 12.66 — no wiring needed
- IQ2 dequant works — Q4_K was the actual root cause

The wasted effort: ~2-3 sessions chasing IQ2 grid tables, merge hash collisions, and imaginary MoE garbage output. The fixed effort: one patch changing 144→176 bytes in the block size.

## True Priority Queue

**P0 — Fix GPU weight loading in bench.c.** Fix bench_e2e zeros → unblocks train_gpu correct forward. This is the #1 thing.

**P1 — Fix train_backprop hang.** Debug why the binary hangs while train_real works. Add fflush stdout, malloc guards. Alternatively: copy the training logic into train_real.c and deprecate train_backprop.c.

**P2 — Add MoE to training.** test_moe passes at 36.6 tok/s. Wire the 256-expert router + shared expert into train_real's forward path.

**P3 — Verify GPU backward pass.** Only worth doing after P0 (GPU forward must give correct CE first).

## What Saved the Project

1. **train_real worked all along.** The CPU forward was correct. The Q4_K fix, once applied, made everything work. The team just didn't run train_real end-to-end and report its output.

2. **The test_moe binary was independently useful.** It proved that MoE was not the source of garbage output, narrowing the search space.

3. **git history preserved the truth.** The commit `fe8c17c Fix Q4_K/Q5_K dequant` was there. The object files were rebuilt. The bug was in *which binary* got tested, not in the code itself.

## Postscript: On False Positives

A benchmark that compares A to B and reports "match within tolerance" is only useful if A produces non-zero output. This seems obvious in retrospect. The fix: always print actual output values in every benchmark, not just the difference. If the max absolute value is 0.000000, something is wrong — mark it FAIL regardless of the match tolerance.

The DA v3 now enforces: **every PASS must show non-zero output values.** No more zero-matches.
