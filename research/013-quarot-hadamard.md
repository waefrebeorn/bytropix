# 013 — Hadamard rotation kills outliers → 4-bit W+A+KV lossless (QuaRot / SpinQuant)

Source: QuaRot (arXiv:2404.00456, 629 cites); SpinQuant (ICLR'25, 487 cites,
learned rotations); QuIP#, GPTQ-Rotate. Also our int8 (B01)/int4 (B02) GEMV.

## Core idea
LLM weight/activation/KV outliers make low-bit quantization hard. A **fixed
Hadamard (or learned) rotation** is a computational-invariant transform: `R·W·Rᵀ`
has the same output as `W` but its *channels are decorrelated* — outliers
spread across all dims, so a uniform int4 quantizer no longer wastes range.
QuaRot fuses R into the weights (free at inference) and applies an online
Hadamard to activations + KV. Result: **4-bit W+A+KV with ~no accuracy
drop, no calibration data needed for RTN**. SpinQuant improves it with *learned*
rotation (R1,R2 merged into weights) — +2 perplexity points at 4-bit.

## Triple-DA
- P1 correctness: rotation is orthonormal ⇒ mathematically identical network.
  Fusing R into W is exact (W' = R·W·Rᵀ). Online Hadamard on x is
  exact given R is known. ✓
- P2 privacy: we compute R ourselves (Hadamard is fixed; learned needs a
  short training pass on local data — no external service). Own C. ✓
- P3 robustness: fixed Hadamard is data-independent (always safe); learned
  SpinQuant needs the rotation fit — we can ship Hadamard-only first (QuaRot
  mode) and add learned later. Degrades gracefully to our existing int8/int4.

## Implementation plan
- `wubu_rotate.c/.h`: `wubu_hadamard(W, rows, cols)` in-place fuse
  (W' = H·W·H via fast Walsh-Hadamard, O(n log n)); `wubu_hadamard_vec(x,n)`
  online. Apply per-linear-input and to K,V before KV-quant.
- Hook into loader: after loading F32 weights, fuse Hadamard into every
  proj weight; store the per-layer input Hadamard size.
- Our int4 GEMV (B02) then quantizes the *rotated* weight → near-lossless 4-bit.

## Test oracle
- Hadamard fuse is exact: `H·W·Hᵀ` matmul vs naive == bit-exact (small n).
- A rotated+int4-quantized Qwen layer vs fp32 reference: cosine >0.995
  (better than non-rotated int4, proving outlier suppression helped).
