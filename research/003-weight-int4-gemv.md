# 003 — int4 weight-only GEMV (Marlin-style packing)

Source: MARLIN (arXiv:2408.11743) — FP16×INT4, 3.9× vs FP16 on A10 for
batch 4–32. AWQ (2306.00978) activation-aware 1% salient channels. Also our
shipped int8 GEMV (B01).

## Core idea
Weights at 4 bits (2 bits/weight actually for symmetric int4 = 16 levels) halve
the weight traffic vs our int8 path. The trick (Marlin) is a *packed, permuted*
layout so the dequant+dot per 8 weights is a small table/shift, keeping the
inner loop compute-light. On a BW-bound decode GEMV this is the next halving after
int8. AWQ shows only ~1% of channels are salient — protecting them (higher
precision or unquantized) preserves accuracy.

## Triple-DA
- P1 correctness: int4 symmetric dequant is exact given per-block scale; cosine
  vs fp32 reference must be >0.99. Salient-channel protection keeps it there.
- P2 privacy: own packing + dequant; no lib. ✓
- P3 robustness: per-block scale prevents outlier blow-up; if a block has a huge
  outlier we fall back to int8 for that block (graceful).

## Implementation plan
- `wubu_gemv_i4(A_i4, scales, x, y, M, K)`: dequant 2 weights at a time via
  `(int8)nybble * scale`, FMA. Pack weights as 2 nybbles/byte in a permuted
  order that keeps 8 consecutive output contributions contiguous (Marlin idea).
- `wubu_gemv_autotune()` adds `use_int4` when M≥256 and K large (BW-bound).
- Wire into `quantized_matmul` F32 path alongside the int8 path.

## Test oracle
Random + real Qwen gate_proj: cosine >0.99 vs fp32 oracle. Assert int4 path is
selected for M=5120 and int8 (not int4) for M=16.
