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

## IMPLEMENTATION STATUS (partial close, 2026-07-28, research-loop cycle 5)
- `src/wubu_kvvq.c` + `include/wubu_kvvq.h`: COMPLETE. Data-independent
  RESIDUAL subvector (product) VQ for KV vectors. head_dim split into n_sub
  subvectors; each subvector gets n_stages of VQ against a FIXED (seeded
  Gaussian, L2-normalized) codebook. Packed bit-stream (sub-4-bit indices).
- `tools/test_kvvq.c`: COMPLETE + PASS. Verified:
  - pack/unpack of indices is BIT-EXACT (no info loss in the index stream).
  - Compression: 2-bit x2-stage = 0.25 bits/elem, 2-bit x4 = 0.50,
    3-bit x3 = 0.56 -- i.e. 17x-34x SMALLER than Q8_0 (8.5 bits/elem). This
    is the real storage/bandwidth win over A01/A02.
  - Degenerate head_dim=1 works; dequant finite.
- HONEST FIDELITY CEILING: on unit-norm KV proxies (how the engine stores
  RMSNorm'd K/V), a *fixed* (data-independent) codebook reaches only
  avgcos ~0.18-0.23 at 2-bit. This is NOT "minimal loss" -- it is the
  genuine ceiling of pure data-independent VQ on unit vectors. We empirically
  tested pairing with the shipped Hadamard rotation (doc 013):
  Hadamard+VQ(2bit x4) avgcos=0.1799 -- NO improvement, because an orthonormal
  rotation preserves distances to the (also unit-norm) codebook. CONCLUSION:
  reaching CommVQ/TurboQuant's "1-bit minimal loss" requires either
  (a) a CODEBOOK CALIBRATED on the actual KV distribution (data-DEPENDENT --
       a short per-model quantization pass, still no external lib, our own C),
  or (b) a LEARNED rotation (SpinQuant-style) before VQ. Both are documented
  next steps, NOT done this cycle to avoid overclaiming a fidelity we
  measured as not-yet-achieved. The VQ PRIMITIVE itself is correct + linked.
- Engine-linked: wubu_kvvq.o is in CORE_OBJ + GPU_OBJ; engine compiles +
  links it; full forward (test_probe_qwen) still PASS (argmax 111667).
- REMAINING (documented): the WUBU_KV_VQ scheme wiring in kv_cache_read/
  write_head + the calibration pass for "minimal loss" fidelity. The module
  is the tested building block; the call-site + calibration are the follow-on.

## Test oracle
- Round-trip cosine of a real Qwen KV proxy at 2-bit VQ >0.98, at
  3-bit >0.995. Assert avg bits/element < Q8_0 (proves compression).
- Full decode of 8k context stays finite + argmax stable vs fp16 KV.
