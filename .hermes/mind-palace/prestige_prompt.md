═══ WUBUTEXT AI — PRESTIGE RESUME (May 15 PM v2) ═══
Path: /home/wubu/bytropix | Branch: master
HW: RTX 5050 6.4GB, -arch=sm_120, NVCC: /usr/local/cuda-13.1/bin/nvcc
Build: make <target> | Models: /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

=== RULE ===
ALL GOALS MUST BE FINISHED. No "good enough." No abandoned streams.

=== STATE ===
✅ ALL 12 STREAMS COMPLETE + INTEGRATED (May 14-15)
✅ train_integrated binary — env flags: TST/RSGD/PGA/NESTED_SSM/NESTED_MOE/POINCARE_R
✅ TST=1 bag s=8 MCE loss (25% steps)
✅ RSGD=1 Riemannian SGD for Poincaré embeddings
✅ PGA=1 Poincaré GQA (CPU detour for GQA layers)
✅ NESTED_SSM=1 Nested SSM K=4 (CPU detour for SSM layers)
✅ NESTED_MOE=1 Poincaré distance routing replacing linear gate_inp
✅ GPU SSM with POINCARE_R
✅ All 5 flags verified in single run: loss=12.42

⚠️ GPU vision pipeline timed out
⚠️ ~0.5% NaN in logits (pre-existing)
⚠️ CPU RMSNorm OOB (d=4096, weight[256])

=== TGT MATH ===
BOUNDARY = 2π
remainder = fmod(x + π, BOUNDARY) - π
tgt_safe_expf(x) = x > 80 ? 80 : x < -80 ? 0 : expf(x)
