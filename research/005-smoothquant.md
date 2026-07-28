# 005 — SmoothQuant activation-outlier migration

Source: "SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs"
(arXiv:2211.10438). NVIDIA quantization concepts blog.

## Core idea
LLM activations have ~1% outlier channels ~100× larger than the rest, making
activation quantization hard. SmoothQuant migrates that difficulty to weights via a
per-channel smoothing factor `s = max(|X|)^α / max(|W|)^(1-α)` (α≈0.5): compute
`Y = (X diag(s)^-1) · (diag(s) W)`. The activation is now smooth (easy to
quantize to int8), the weight is slightly harder but still easy. This unlocks
**int8 activation × int8 weight** GEMV (vs our current int8-weight × fp32-act),
halving activation traffic too — the next bandwidth win after B01.

## Triple-DA
- P1 correctness: the transform is mathematically equivalent (Y = X·W). Calibration
  needs a small sample set to estimate per-channel max — we can use a fixed
  calibration prompt or the model's own first prefill pass.
- P2 privacy: calibration on local data; no external service. ✓
- P3 robustness: α is a tunable knob; if a layer is outlier-free, s≈1 and we get
  vanilla int8. Degrades gracefully.

## Implementation plan
- `wubu_smooth_quant.c/.h`: estimate per-channel activation max from a calibration
  pass, compute `s`, produce smoothed weight `W' = diag(s)·W` and store `s` for
  the activation side.
- Extend `wubu_gemv_i8` to take int8 activations (pre-scaled) → full int8×int8
  GEMV with int32 accumulate.
- Autotune: enable when both weight and activation fit int8 comfortably.

## Test oracle
Compare int8×int8 smoothed GEMV vs fp32 reference on real Qwen layers: cosine
>0.99 (SmoothQuant claims near-lossless). Assert activation bytes halved vs B01.
