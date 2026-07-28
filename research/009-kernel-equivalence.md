# 009 — Bounded equivalence testing of quantized kernels

Source: "Equivalence Checking of ML GPU Kernels" (2511.12638); Gimlet "Formally
Verifying AI-Generated Kernels"; Alive2 (PLDI'21, bounded translation validation,
LLVM); USENIX'25 HEC (equivalence via equality saturation); seL4 binary-level
verification.

## Core idea
We currently validate quantized GEMV with a *single* cosine check. That's weak —
one random tensor can hide bugs. Borrow the Alive2 / bounded-translation-validation
mindset: prove `quantized_gemv(W,x) ≈ reference_gemv(W,x)` over a *bounded but
exhaustive* input space, and add **invariant** checks (output finite, scale
non-zero, no NaN in dequant). The "bounded" part means we test over small M/K with
exhaustive weight/activation bit patterns, not just one random draw.

## Triple-DA
- P1 correctness: this IS a correctness tool. Catches the class of bug we hit in
  the int8 GEMV (broken dot-product math) automatically. ✓
- P2 privacy: local test harness. ✓
- P3 robustness: bounded = fast; scale it to the real M/K as a sampled fuzz. Never
  blocks the build (runs as a test target, not in hot path).

## Implementation plan
- Add `tools/test_gemv_equivalence.c`: for M,K in {1,4,16,64}, exhaustively or
  heavily-sample weight values in {-1,0,1, small ints} and x in a few patterns;
  assert `quant_gemv` == `scalar_gemv` within tolerance, AND invariants
  (finite, no NaN). Run for int8, int4, ternary paths.
- Optional: an SMT-backed checker later (Alive2-style) — out of scope for now.

## Test oracle
- The test file itself: must catch an *injected* bug (we temporarily break the
  int8 inner loop and assert the test fails). Self-validating.
