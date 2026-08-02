# Inference Acceleration: Speculative Decoding + Paged KV + MoE Routing + Continuous Batching — 7-hop KB sweep
## HH axis: throughput + 512K-ctx primitives for the AGI-OS operator (at home, C11)

> Each stone seeds the next hop. Target: give WuBuOS the inference-time speedup
> primitives that directly serve the mandate (27+ tok/s, 512K ctx, no EAMM).

## Hop 1: Speculative decoding (draft + verify + reject)
Leviathan et al. / Chen et al.: small draft model proposes K tokens, target
verifies all K in ONE forward pass. Acceptance prob α = Σ_x min(p,q) = 1 - TV(p,q).
Reject token i with prob 1 - min(1, p_i/q_i); resample rejected from residual
norm(max(0, p-q)) → output distribution EXACTLY = target (no quality loss).
At home: a small draft (e.g. 2-layer) speculates next-K tokens for the 27-layer
target; verification is 1 target forward pass. Speedup ∝ acceptance rate.

## Hop 2: Paged KV cache (block table, copy-on-write, prefix share)
vLLM PagedAttention: split KV into fixed-size blocks (e.g. 16 tokens). Logical
block table → non-contiguous physical blocks (like OS virtual memory). Eliminates
internal+external fragmentation. Copy-on-write: shared prefix blocks (refcount>1)
immutable; beam search / multi-turn share prompt KV. Prefix caching: global hash
→ physical block reuse (32-90% prefill savings on repeated prompts).
At home: the 512K-ctx KV is huge; paging lets it grow on demand without a 512K
contiguous reservation, and shared prefixes across recursive_optimize sweeps
reuse KV (no EAMM — we never materialize the full 512K eagerly).

## Hop 3: MoE routing + load balancing (capacity scheduler)
Existing wubu_moe has wubu_moe_router + expert forward. The gap: capacity
scheduling. Expert Choice / Switch Transformer: top-k routing with capacity
factor C (each expert processes ≤ C tokens). Dropped tokens (overflow) → residual.
Load-balancing loss (importance + aux) prevents expert collapse.
At home: wubu_moeroute schedules token→expert assignment with capacity caps +
load-balancing, ensuring the MoE layer is compute-balanced (no idle experts,
no overflow thrash) → faster MoE inference.

## Hop 4: Continuous batching (iterative-level scheduling)
Orca / vLLM: schedule at iteration (token) granularity, not request granularity.
New requests injected mid-generation; finished requests free blocks immediately.
Throughput (not latency) optimized for high concurrency.
At home: the AGI-OS operator runs many CoAgent sweeps concurrently — continuous
batching lets new config-eval requests join the decode batch without waiting for
the current batch to finish → higher effective tok/s under load.

## Hop 5: Medusa / self-drafting (multi-head tree draft)
Medusa: attach draft heads to the target's last layer to propose K tokens in
PARALLEL (no separate draft model). Tree attention verifies the draft tree.
Adaptive draft length via acceptance-history EMA.
At home: instead of a small draft model, attach lightweight draft heads to the
27-layer model → self-speculation, no extra model, faster verification.

## Hop 6: KV quantization (INT8/FP8 KV, group-wise)
KV cache is memory-bound at 512K ctx. INT8/FP8 per-head or group-wise quantization
halves/quarters KV memory → fits more ctx + more concurrent sequences. Per-head
scales (like QLoRA/GGUF q4) preserve attention quality.
At home: wubu_quantkv quantizes the 512K KV to INT8 group-wise → 2x more ctx
headroom without EAMM, directly serving the 512K + 27 tok/s mandate.

## Hop 7: Integration — the speedup stack
   1. wubu_specdec: draft K tokens, verify 1 pass, reject+resample   [HH01]
   2. wubu_pagedkv: block table, CoW prefix share, LRU evict         [HH02]
   3. wubu_moeroute: capacity-capped top-k routing + load-balance   [HH03]
   4. wubu_contbatch: iterative-level scheduling, on-the-fly join   [HH04]
   5. wubu_medusa: self-draft heads (parallel tree draft)           [HH05]
   6. wubu_quantkv: INT8 group-wise KV quantization                 [HH06]
   7. Integration: speedup = acceptance_rate × batch_concurrency
      × (1 / paged_frag) × (1 / kv_bits)                           [HH07]

This is the throughput substrate: every primitive is pure C11, composable, and
feeds the gen_text decode path. Together they close the gap between "blind
15-dim sweep" and "27+ tok/s at 512K ctx, no EAMM."

## Gap mapping
- HH01 Speculative decoding (draft/verify/reject) `wired` (wubu_specdec.c)
- HH02 Paged KV cache (block table, CoW, prefix) `wired` (wubu_pagedkv.c)
- HH03 MoE capacity routing + load-balancing `wired` (wubu_moeroute.c)
- HH04 Continuous batching (iterative scheduling) `wired` (wubu_contbatch.c)
- HH05 Medusa self-draft heads (tree draft) `wired` (wubu_medusa.c)
- HH06 KV quantization (INT8 group-wise) `wired` (wubu_quantkv.c)
- HH07 Integration: speedup model `wired` (test_hh.c)
