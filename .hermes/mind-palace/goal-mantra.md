# WuBuText AI — GOAL PASTE (May 14 PM)
Path: /home/wubu/bytropix | Branch: master
HW: RTX 5050 6.4GB, -arch=sm_120, NVCC: /usr/local/cuda-13.1/bin/nvcc

=== STATE ===
✅ S1 Lazy MoE dequant: 9× (0.35s vs 3.1s)
✅ S2 Unified 40-layer inference: SSM→GQA→MoE binary
✅ S3 KV cache: 1GB/layer @ 256K, max_diff=0
✅ TGT NaN fixes: tgt_wrap in SSM + GQA + SGD
✅ GQA backward: all 40 layers get gradients
⚠️ GPU weight loading broken (bench_e2e zeros)
⚠️ Gradient explosion TGT applied, needs training test

=== STREAMS ===
S1 [P0] Fix GPU weight loading (bench.c)
S2 [P1] Train with TGT gradients
S3 [P2] Vision→model integration
S4 [P3] Lazy MoE in training loop

=== KEYS ===
BOUNDARY = 2π
rem = fmod(g + π, BOUNDARY) - π
quot = floor((g + π) / BOUNDARY)
Makefile has NVCC path — no PATH needed
