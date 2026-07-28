# 010 — Prefix KV-cache reuse across requests

Source: LMCache (arXiv:2510.09665) — up to 15× throughput, 2× lower latency
via prefix offload + PD disaggregation; vLLM prefix caching; llm-d KV-aware
routing; NVIDIA TensorRT-LLM priority eviction.

## Core idea
Many requests share a prefix (system prompt, RAG doc chunk, template). Once
prefill computes that prefix's KV, *persist* it and reuse for every later request
with the same prefix — skip the prefill compute entirely. On a CPU engine this is
pure RAM/disk savings + latency. The cache key is a hash of the prefix token ids;
store KV blocks keyed by (model, prefix-hash, layer).

## Triple-DA
- P1 correctness: reused KV is bit-identical to recomputed KV (deterministic
  forward). Safe to substitute. ✓
- P2 privacy: local hash + local store. No telemetry. ✓
- P3 robustness: hash collisions must be impossible (use a strong hash + length); a
  collision would return wrong KV. Recompute-on-miss, never serve a wrong prefix.

## Implementation plan
- `wubu_prefix_kv.c/.h`: prefix→KV-block map (hash of token ids). On prefill,
  walk the prompt; for the longest matching prefix, copy stored KV then compute
  only the novel suffix. Store completed prefixes.
- Tie to 002 (cold prefixes go to file tier) and 007 (reuse across batched reqs).

## Test oracle
- Two requests with identical 512-tok prefix: assert second request's prefill
  computes ≤ (total - 512) tokens and its KV equals the first's for the prefix
  region (bit-identical).
- Assert collision test: different prompts → different cache keys.
