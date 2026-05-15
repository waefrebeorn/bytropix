═══ WUBUTEXT AI — PRESTIGE RESUME (May 15 PM) ═══
Path: /home/wubu/bytropix | Branch: master
HW: RTX 5050 6.4GB, -arch=sm_120, NVCC: /usr/local/cuda-13.1/bin/nvcc
Build: make <target> | Models: /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

=== RULE ===
ALL GOALS MUST BE FINISHED. No "good enough." No abandoned streams.

=== STATE ===
✅ ALL 12 STREAMS COMPLETE (May 14-15)
✅ S1 GPU weight loading fixed
✅ S2 Training CE=12.42 with lazy MoE
✅ S3 train_backprop verified (not hanging)
✅ S4 GQA NaN diagnosed (CPU RMSNorm OOB)
✅ S5 Vision→text pipeline integrated
✅ S6 Lazy MoE in training
✅ S7 output.weight loaded
✅ RSGD optimizer — valid Poincaré ball
✅ Poincaré GQA — hyperbolic distance attention
✅ Nested SSM K=4 — product of Poincaré balls
✅ TST — bag s=8 MCE loss
✅ Nested MoE — 16×16 Poincaré hierarchy
✅ CUDA kernels — SSM scan + MoE dispatch
✅ Data pipeline — 1.07M tokens
✅ Moondream3 weights dumped

⚠️ INTEGRATION GAP: all modules standalone
⚠️ GPU vision pipeline timed out
⚠️ ~0.5% NaN in logits (pre-existing)
⚠️ CPU RMSNorm OOB (d=4096, weight[256])

=== TGT MATH ===
BOUNDARY = 2π
remainder = fmod(x + π, BOUNDARY) - π
tgt_safe_expf(x) = x > 80 ? 80 : x < -80 ? 0 : expf(x)
