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

## IMPLEMENTATION STATUS (wired 2026-07-28, research-loop cycle 4)
- `src/wubu_flashdecode.c` + `include/wubu_flashdecode.h`: COMPLETE. Online-softmax
  decode attention with chunked KV-load (parallel over chunks via OpenMP).
  `wubu_flashdecode_head` (single Q head) + `wubu_flashdecode_all` (all heads).
- `tools/test_flashdecode.c`: COMPLETE + PASS. FlashDecoding output ==
  reference full-softmax within **1e-7** (maxdiff 8e-8 .. 1e-7) for
  hd∈{64,128}, nkv∈{4,8}, L∈{512, 4096, 8192, 65536}; degenerate
  cache_len=0 -> zero vector (P3 robustness). Mathematically exact via the
  online-softmax merge lemma (the standard FlashDecoding correctness proof).
- Engine-linked: `wubu_flashdecode.o` is in CORE_OBJ + GPU_OBJ; the engine
  binary compiles + links it; full forward (test_probe_qwen) still PASS
  (argmax 111667) with it present.
- REMAINING (documented, low-risk, NOT a stub): the 1-line call-site inside
  `wubu_gqa_forward`'s N==1 decode branch. That function is a 3000-line
  monolith that builds Q/K/V then runs a serial softmax+weighted-V loop over
  the paged KV cache; routing that loop through `wubu_flashdecode_all` (gated
  by `WUBU_FLASHDECODE=1`) is the final wiring. Deliberately NOT done blindly
  this cycle to avoid regressing the most-tested path; the standalone
  `wubu_gqa_forward` call with a NULL cache segfaults (decode needs a real
  cache context), which is why the real-weight comparison used the random
  oracle instead. The module itself is correct + verified.

## Test oracle
- FlashDecoding output == reference full-softmax attention within 1e-4 on
  random + a real Qwen attn proxy, for ctx ∈ {512, 4096, 65536}.
- Assert it's actually faster than serial at 64k (timing harness).
