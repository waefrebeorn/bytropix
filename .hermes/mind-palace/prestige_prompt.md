═══ WUBUTEXT AI — PRESTIGE RESUME (May 15) ═══
Path: /home/wubu/bytropix | Branch: master
HW: RTX 5050 6.4GB, -arch=sm_120, NVCC: /usr/local/cuda-13.1/bin/nvcc
Build: make <target> | Models: /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

=== RULE ===
ALL GOALS MUST BE FINISHED. No "good enough." No "works today." No abandoned streams.

=== STATE ===
✅ S1 GPU weight loading: 626c143 — dequant bypass in unbuffered reads fixed
✅ S2 Training pipeline: ac8b81c — GPU forward + TGT gradients, CE=12.42
✅ S3 train_backprop: verified not hanging (just CPU-slow 25s/step)
✅ S4 GQA L3 NaN: CPU RMSNorm dim mismatch (d=4096 wt weight[256]), GPU OK
✅ S5 Vision→text pipeline: 1e15f8a — real screenshot, 0 NaN
✅ S6 Lazy MoE in training: 03674c6 — top-8/256 cached fwd/bwd
✅ S7 output.weight from GGUF: already loaded

⚠️ Pre-existing: ~0.5% NaN in model logits (any input source)
⚠️ CPU GQA RMSNorm: d=4096 with weight[256] — OOB read for i>=256

=== REMAINING WORK (post-S7) ===
Math: RSGD optimizer, Poincaré GQA, nested SSM, nested MoE
Manifold: Moondream3 port, Poincaré distance router
Optimizations: GPU vision pipeline, data pipeline, TST, CUDA kernels
Bugs: model logit NaN, CPU RMSNorm

=== TGT MATH ===
BOUNDARY = 2π
remainder = fmod(x + π, BOUNDARY) - π   # [-π, π], preserves direction
tgt_safe_expf(x) = x > 80 ? 80 : x < -80 ? 0 : expf(x)
