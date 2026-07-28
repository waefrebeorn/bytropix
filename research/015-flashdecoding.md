# 015 — Parallel KV-load decode attention (FlashDecoding)

Source: FlashDecoding (Dao et al., PyTorch blog / Stanford CRFM, 2023) — up to
8× faster decode for long sequences; FlashDecoding++ (MLSys'24) 4.86×.
Also FlashForge (prefix-aware). Our paged KV + KV-quant paths.

## Core idea
Standard decode attention loads the whole KV cache serially per query, then
does one big softmax. FlashDecoding adds a **new parallel dimension: the
KV sequence length** — split K/V into chunks, compute partial attention
(softmax over each chunk, keep the running log-sum-exp + partial output)
in parallel, then a tiny final reduction combines them. For long contexts
this parallelizes the otherwise-serial KV read, cutting decode attention
latency ~8× at 100k+ tokens. Maps onto our already-chunked paged KV.

## Triple-DA
- P1 correctness: the chunked softmax + LSE reduction is mathematically
  identical to full softmax (standard flash-attention online-softmax lemma). ✓
- P2 privacy: pure kernel math, own C. ✓
- P3 robustness: chunk size is a tunable; tiny contexts (≤512) skip the
  split (serial is fine). No numerical cliff (LSE accumulation is stable).

## Implementation plan
- `wubu_attn_flashdecode.c/.h`: given q (1 vec), K/V pages (chunked),
  for each chunk compute local max/sumexp + partial `Σ exp·v`; combine
  across chunks with running m (max) and l (log-sum-exp) — final
  `out = Σ exp(s_i - m)·v_i / exp(l)`. Vectorize the chunk inner loop.
- Hook into the decode attention path (currently serial paged read).

## Test oracle
- FlashDecoding output == reference full-softmax attention within 1e-4 on
  random + a real Qwen attn proxy, for ctx ∈ {512, 4096, 65536}.
- Assert it's actually faster than serial at 64k (timing harness).
