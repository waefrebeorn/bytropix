# 011 — Attention-sink-free gated attention (kills massive activations)

Source: "The Spike, the Sparse and the Sink" (arXiv:2603.05498); Qiu et al.
"Gated Attention for LLMs: non-linearity, sparsity, attention-sink-free" (NeurIPS
2025); Qwen3-Next adopts gated attention for stability.

## Core idea
Massive activations (a few dims ~100× larger) and attention sinks (first token
gets disproportionate attention) are *learned* artifacts. Conditional gating —
`gate = σ(W_gate(x))` applied per-channel/per-head *dynamically* — suppresses
sinks and eliminates the need for the "no-op" sink token. Why this is an
*engine* concern: sinks force the KV cache's first row to stay hot (can't evict,
ties to 002), and massive activations make activation quantization hard (ties to
005 SmoothQuant). Supporting gated attention in the forward lets sink-free models
run leaner.

## Triple-DA
- P1 correctness: gating is a multiplicative mask on attention output; exact given
  the gate. Only relevant for models trained with it (Qwen3-Next family). ✓
- P2 privacy: pure math. ✓
- P3 robustness: unconditional/static gating *fails* to suppress sinks (per the
  paper) — so we only enable dynamic per-channel gating, never static.

## Implementation plan
- Add `ATTN_GATED` layer variant: after softmax·V, multiply by `σ(W_gate·x)`,
  then out-proj. Mirror Qwen3-Next's exact gating order.
- Loader detects the `W_gate` weight presence → marks layer gated.

## Test oracle
- Gated layer reproduces reference PyTorch GatedAttention (cosine >0.99).
- Assert sink-ratio (attention mass on token 0) drops vs ungated baseline on a
  gated model.
