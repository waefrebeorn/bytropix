# 001 — KV entropy-aware adaptive compression (Ecco, ISCA'25)

Source: Cheng et al., "Ecco: Improving Memory Bandwidth and Capacity for LLMs via
Entropy-Aware Cache Compression", ISCA 2025 (dl.acm.org/10.1145/3695053.3731024).
Also: Predictive Multi-Tier (2604.26968), MTDS (s40747-025-02200-4).

## Core idea
Not every KV block needs the same bit-width. Ecco measures per-block entropy and
assigns 2–8 bits adaptively, storing the chosen width per block. Result: up to
2.9× speedup over AWQ, ~4× capacity, SOTA accuracy. The bandwidth win is that
cold/low-entropy blocks move half (or quarter) the bytes on every decode read.

## Triple-DA
- P1 correctness: entropy→bitwidth is a pure function; dequant is exact given the
  stored scale+width. No accuracy cliff if width is bounded by a max (we cap 8).
- P2 privacy: pure local math, no external lib. OWN-C. ✓
- P3 robustness: a single block at 2-bit can drift; mitigate by keeping a *minimum*
  width floor (e.g. 4-bit for the first 4 layers where attention is sharp) and
  per-block scale. Degrades gracefully to Q8_0 when entropy high.

## Implementation plan (builds on wired A01/A02)
- Extend `wubu_kvcache_quant.h` with `wubu_kvq_adaptive_quant(z, out, width_bits,
  n)` that picks block bit-width from block variance (proxy for entropy): low var→
  2-4bit, high var→8bit. Store `(width, scale, packed)`.
- Add `wubu_kv_scheme_t` value `WUBU_KV_ADAPTIVE`; `wubu_kv_select()` enables it
  when s (context) is huge AND variance spread across layers is high.
- `kv_cache_write_head` packs with chosen width; `kv_cache_read_head` unpacks.

## Test oracle
Same as test_kvcache_quant: cosine of round-trip vs fp16 must be >0.995 for a
random block, and >0.99 for a real Qwen attn-proj KV proxy. Assert the *average*
bits/element is <8 when entropy is low (proves compression actually engaged).
