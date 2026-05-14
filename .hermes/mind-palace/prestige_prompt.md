── WuBuText AI — PRESTIGE PROMPT (May 14 PM) ──
Path: /home/wubu/bytropix | Branch: master (15 commits ahead)
HW: RTX 5050 6.4GB, -arch=sm_120, NVCC: /usr/local/cuda-13.1/bin/nvcc

=== STATE (inference phase complete) ===
✅ S1 Lazy MoE dequant — infer_moe_lazy: 0.35s vs 3.1s (9×). Only top-8/256 experts.
✅ S2 Unified 40-layer inference — infer_unified: SSM→GQA→MoE chain.
✅ S3 GQA KV cache design — test_kv_cache: max_diff=0, 1GB/layer @ 256K.
✅ S5 Mind palace — updated.
✅ TGT NaN/Inf fixes — tgt_wrap + tgt_safe_expf in SSM state, GQA scores, SGD step.
✅ GQA backward wired — all 40 layers get real gradients.
✅ Poincaré GPU SSM — 2835 tok/s.
✅ GPU vision — 217ms/256x256 (161× speedup).
⚠️ P0 [BROKEN] GPU weight loading — bench_e2e all zeros, train_gpu CE 69 vs 12.66.
⚠️ P0 [NaN] GQA L3 corrupted inputs — memory hypothesis: MoE load overwrites GQA input buf.
⚠️ P0 [TRAINING] gradient explosion — TGT wrapping applied to SGD, needs training test.

=== CRITICAL GAPS ===
G1 [P0] GPU weight loading — bench.c gpu_load_ssm_layer produces zeros.
G2 [P1] train_backprop hangs at model init.
G3 [P2] Vision→model integration (mmproj → 40-layer pipeline).
G4 [P3] Mind palace stale (now fixed).

=== BUILD ===
make <target> — NVCC path in Makefile (no PATH needed).
