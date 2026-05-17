# WuBuText AI — Overnight Navigation Map (May 17 v3 — SSM verified, TGT/manifold cleared)

## Where We Are
All mathematical components verified against llama.cpp. TGT/manifold math is NOT in the inference path. Hidden state is still orthogonal (cos-sim 0.0167). The bug is below the abstraction level of code reading — must dump layer-by-layer.

## What Changed This Session
- In-depth comparison of all 14 SSM/GQA/math components vs llama.cpp
- TGT audit: `tgt_wrap` only in `wubu_gqa_forward` (not called by infer_text.c)
- Manifold audit: `wubu_poincare_ssm_forward` not in inference path
- Verified `ssm_a` values: all negative, range -72 to -0.019 — correct for DeltaNet
- Gate values never approach ±80, so `tgt_safe_expf` clamping never triggers
- Confirmed layer mapping via llama-simple: layers 3,7,11,15,19,23,27,31,35,39 = GQA

## Build
rm -f src/cuda_kernels.o infer_text; make infer_text  # full rebuild

## Next Step
Layer-by-layer dump comparison. Build dump points in both engines, find first divergent layer.
