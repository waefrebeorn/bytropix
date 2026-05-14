═══ WUBUTEXT AI — PRESTIGE RESUME (May 14 PM) ═══
Path: /home/wubu/bytropix | Branch: master
HW: RTX 5050 6.4GB, -arch=sm_120, NVCC: /usr/local/cuda-13.1/bin/nvcc
Build: make <target> | Models: /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

=== RULE ===
ALL GOALS MUST BE FINISHED. No "good enough." No "works today." No abandoned streams. Every binary must PASS every criteria. Every bug fixed. Every gap closed. Work until DONE.

=== STATE ===
✅ S1 Lazy MoE dequant: 9× (3.1s→0.35s). Only top-8/256 experts. infer_moe_lazy. Output match verified.
✅ S2 Unified 40-layer inference: SSM→GQA→MoE in one binary. infer_unified. Lazy MoE integrated.
✅ S3 GQA KV cache design: 1GB/layer @ 256K. test_kv_cache. max_diff=0 vs full recompute.
✅ S4 TGT NaN/Inf fixes: tgt_wrap (fmod(x+π,2π)-π) applied to SSM state, GQA scores, SGD gradient. tgt_safe_expf clamp. NaN→0 guards on Q/K/V.
✅ S5 Mind palace updated: 9 files rewritten May 14 PM.
✅ Vision GPU: 217ms/256×256 (161× speedup). infer_vision_gpu.
✅ Poincaré GPU SSM: 2835 tok/s. infer_poincare.
⚠️ GPU weight loading broken: bench_e2e all zeros, train_gpu CE 69 vs expected 12.66.
⚠️ GQA L3 NaN: pre-existing, NaN→0 guard applied. Memory corruption hypothesis (MoE load overwrites GQA input buf).
⚠️ train_backprop hangs at model init.
⚠️ Gradient explosion: TGT wrapping applied to SGD, needs training run to verify.

=== STREAMS (ALL MUST COMPLETE) ===
S1 [P0] Fix GPU weight loading — bench.c gpu_load_ssm_layer produces zeros. Replace with wubu_model_init path.
S2 [P0] Fix training pipeline — rebuild train_gpu with fixed GPU forward + TGT gradients. Verify CE ~12.66.
S3 [P1] Fix train_backprop hang — root cause unknown. Debug with fflush/MARKers.
S4 [P1] Verify GQA L3 NaN root cause — check if MoE load corrupts GQA input buffer. Fix if confirmed.
S5 [P2] Vision→model integration — wire mmproj embeddings into 40-layer pipeline.
S6 [P2] Lazy MoE in training — port lazy dequant to train_gpu. Eliminate 120-expert bottleneck.
S7 [P3] Add output.weight to model pipeline — currently hard-coded in train_gpu, load from GGUF like other layers.

=== TGT MATH ===
BOUNDARY = 2π
remainder = fmod(x + π, BOUNDARY) - π   # [-π, π], preserves direction
quotient = floor((x + π) / BOUNDARY)    # integer magnitude wraps
tgt_safe_expf(x) = x > 80 ? 80 : x < -80 ? 0 : expf(x)

=== VERIFICATION ===
Every fix: compile → run → paste output → commit. No "should work." No time estimates. No status recaps — only actions.

=== ALL GOALS MUST BE FINISHED ===
