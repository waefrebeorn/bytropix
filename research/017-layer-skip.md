# 017 — Token-wise layer skipping (GateSkip / LayerSkip / Mixture-of-Depths)

Source: GateSkip (2025, residual-stream gating, differentiable); LayerSkip
(early-exit + self-speculative decoding); Mixture-of-Depths (MoD, Raposo
2024) router at every layer; "Learning to Skip the Middle Layers".
Also our Gated-DeltaNet (008) and MoE (wubu_moe).

## Core idea
Not every token needs all L layers. A learned (or heuristic) **per-token gate**
decides to skip a layer's compute: `y = x + gate·F(x)` where `gate≈0`
*skips* the layer (just passes the residual). MoD shows 21% faster with
minimal quality loss; LayerSkip exits early for easy tokens then lets the
remaining layers verify. This is **adaptive compute** — the complement to
our fixed-quantization speedups: quantize *every* layer (B-series) AND
skip *some* layers for *some* tokens.

## Triple-DA
- P1 correctness: gate=1 reproduces the full layer; gate=0 is a clean
  skip (residual passthrough). Only meaningful for models trained with the
  gate (Qwen3-Next family, MoD-trained). For untrained models a
  *heuristic* gate (skip a layer if input norm < threshold) is a safe
  approximation we can ship first. ✓
- P2 privacy: gate is local math. ✓
- P3 robustness: never skip *all* layers (keep a floor, e.g. last 4
  layers always run). A wrong skip degrades quality, not correctness —
  and LayerSkip's self-speculative verify catches it.

## Implementation plan
- `ATTN_GATESKIP` layer variant: after computing F(x), `y = x + σ(g)·F(x)`
  (g from a small gate weight). Loader detects the gate weight → marks layer.
- Heuristic fallback (no gate weight): skip layer i if `||x|| < τ_i` (τ
  learned offline or a constant); floor = always-run last 4 layers.
- Tie to scheduler (007): a skipped layer just advances the token.

## Test oracle
- Gated layer with σ(g)=1 == ungated reference (cosine 1.0); σ(g)=0
  == input passthrough (proves clean skip).
- Assert floor respected (last 4 layers never skipped) on a generation run.
