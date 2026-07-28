# 014 — Sub-4-bit KV via data-independent vector quantization (CommVQ / TurboQuant)

Source: CommVQ (Apple, ICML'25, arXiv:2506.18879) — 1-bit KV with
minimal loss, 87.5% smaller; TurboQuant (Google, ICRL'26, arXiv:2504.19874)
3-bit, zero accuracy loss, **data-independent codebooks** (online, no training).
Also our KIVI (A02) and entropy Ecco (A03).

## Core idea
Push KV cache below 4 bits using **vector quantization with precomputed,
data-independent codebooks** (PolarQuant + QJL style). The killer property:
the codebook is *not* trained on the data distribution, so you quantize each K/V
vector **as it arrives** during autoregressive decode — zero offline training,
zero reindexing. CommVQ hits 1-bit KV at minimal loss; TurboQuant
3-bit lossless. This is the next 2–4× on top of our KIVI/Q8_0 KV path
and, unlike K-means PQ, needs no calibration.

## Triple-DA
- P1 correctness: VQ with a fixed codebook is a pure lookup+reconstruct;
  error bounded by codebook granularity. 1-bit is aggressive (use for
  cold/long-context layers only). ✓
- P2 privacy: codebook is a constant table we ship; no external lib. Own C. ✓
- P3 robustness: data-independent ⇒ no calibration drift. Fallback to
  Q8_0 (A01) for layers where 1-bit drifts (early layers, sharp attention).

## Implementation plan
- `wubu_kvq_vq.c/.h`: precomputed codebook (e.g. 256 entries of
  head_dim-vectors from a fixed polar/Gaussian init). `wubu_kvq_vq_quant`
  finds nearest codebook entry per K/V vector (sub-sampled search for speed);
  `wubu_kvq_vq_dequant` returns the codeword. Store codebook idx (1-2 bits).
- Add `WUBU_KV_VQ` scheme to `wubu_kv_select`; enable when s (ctx)
  is huge (>=32768) — same trigger as KIVI but beats it on bits.

## Test oracle
- Round-trip cosine of a real Qwen KV proxy at 2-bit VQ >0.98, at
  3-bit >0.995. Assert avg bits/element < Q8_0 (proves compression).
- Full decode of 8k context stays finite + argmax stable vs fp16 KV.
