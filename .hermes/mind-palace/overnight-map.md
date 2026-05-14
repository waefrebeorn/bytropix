# WuBuText AI — Overnight Navigation Map (May 13 DA Update)

## Where We Are

**DA Audit complete.** 8 binaries tested. 5 pass, 3 fail in new ways.

### Verified ✅
- train_real: CE 12.66, forward pass correct on CPU
- test_fused_vs_old: GPU diff 0.036
- test_tokenizer: CJK round-trip works
- test_moe: [-0.028, 0.031], NaN=0, 36.6 tok/s
- dump_mmproj: 334 tensors

### Broken ⛔ (NEW findings — not in old docs)
- bench_e2e: ALL ZEROS output (was previously claimed as "PASS: 29x speedup")
- train_gpu: CE loss 69 vs expected 12.66
- train_backprop: hangs at 180s

### Critical Context
- Q4_K dequant fix resolved ALL NaN and garbage output issues
- Old prestige prompt claims about "IQ2 garbage" and "CE loss commented out" are FALSE
- GPU weight loading in bench.c is the ROOT CAUSE of bench_e2e + train_gpu failures

## Workstreams

### A — Fix GPU weight loading (P0)
Root cause of bench_e2e zeros + train_gpu wrong loss. `gpu_load_ssm_layer()` reads bad data from GGUF.

### B — Fix train_backprop hang (P1)
Same code as train_real. Add fflush + malloc guards.

### C — Integrate MoE into training (P2)
test_moe passes. Wire 256 experts into train_real.

## Data You Should Not Re-Derive
- train_real is the ONLY correct forward path — uses wubu_model_forward_from_embd with pre-loaded CPU weights
- bench_e2e reopens GGUF 40 times per run — old code path from before Q4_K fix
- train_gpu reopens GGUF 40 times PER STEP — also broken weight loading
- All PASS claims from prior sessions about bench_e2e are based on zero-output or stale buggy weights

## Fallback
If all workstreams stuck: clean rebuild from scratch, then test train_real alone. That path works.
