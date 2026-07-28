# 008 — Gated-DeltaNet 3:1 hybrid linear attention

Source: Qwen3-Next / Kimi Linear (3:1 Gated DeltaNet : full attention); Raschka
LLM Architecture Gallery "Gated Attention"; arXiv Gated-DeltaNet refs. DeepSeek-V4
mHC (2512.24880) for latent-attention context.

## Core idea
Replace most attention layers with a *linear* (recurrent) mixer — Gated DeltaNet —
that keeps a fixed-size state `S` updated by a delta rule, scaling **linearly** not
quadratically with context. Keep ~1 in 4 layers as full attention for exact
retrieval. The KV cache for the linear layers is just the small state `S` (head_dim²
per head) — essentially free vs the quadratic KV of full attention. This is the
biggtest *architectural* KV saving possible.

## Triple-DA
- P1 correctness: the delta rule `S = a·S + k·(v - S·k)·β` is exact; needs
  QK-norm (L2) and a sigmoid gate + RMSNorm + SiLU out-gate (stabilizes, removes
  attention sinks — ties to A09). Our SSM scan (wubu_ssm_scan) is the prefill
  primitive; the recurrence is the decode primitive.
- P2 privacy: pure math. ✓
- P3 robustness: linear attention compresses context into `S`; for very long
  contexts retrieval degrades — hence the 3:1 full-attention holdout. We support
  both and let the loader pick the per-layer type from the model config.

## Implementation plan
- Extend the model config to mark each layer as `ATTN_FULL` / `ATTN_DELTANET`.
- `wubu_deltanet_step()` (recurrence, decode) + `wubu_deltanet_prefill()`
  (chunkwise, reuses wubu_ssm_scan). Out-gate = RMSNorm(SiLU(gate)).
- The KV cache for deltanet layers stores `S` (tiny), not full K/V.

## Test oracle
- Identity test: a deltanet layer on a random input reproduces a reference
  PyTorch GatedDeltaNet implementation (cosine >0.99).
- Assert deltanet KV footprint ≪ full-attention KV footprint at 32k context.
