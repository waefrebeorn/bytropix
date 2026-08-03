# 045 — Expanded tokenization for simpler depth (BLT)

> Status: `open`. Date: 2026-08-03.
> Sources: BLT arXiv:2412.09871 (Pagnoni et al., FAIR Meta);
> Multi-token prediction arXiv:2404.19737 (Gloeckle et al.);
> BPE-dropout ACL2020 (Provilkov et al.).

## The one-sentence idea

WuBu learns from raw BYTES grouped into dynamic PATCHES (entropy-gated),
so the main transformer runs once per patch — 4–8× fewer global steps —
which means the SAME work needs a SIMPLER (shallower) body: patch size
becomes a grow/shrink axis alongside depth.

## Why the seed can use this

The 35M seed (12 layers, dim 448, byte-level BPE vocab 16384) spends its
entire depth budget on subword structure. BLT shows byte-level modeling
with dynamic patching matches tokenizer models flop-controlled (up to 50%
fewer inference flops at Llama-3 scale) and is MORE robust to noise and
long-tail data. The win transfers to small models: the byte modules
(local encoder/decoder) are tiny; the expensive global transformer runs
fewer times.

## The architecture (BLT §2-3)

1. **Entropy model**: a tiny byte-LM (or 2-byte-context CNN per BLT §2.3f)
   scores next-byte entropy. High entropy → new patch starts (short
   patches = more compute for hard bytes); low entropy → long patches
   (predictable bytes are cheap). Constraints: reset entropy context at
   newlines; approximate monotonicity (§4.4) to stop drift.
2. **Local encoder** (§3.2): per-byte embeddings + HASH n-gram embeddings
   (no learned patch vocabulary — a fixed hash table, no OOV) + pooling
   per patch + a few encoder layers + cross-attention from patch reps to
   byte reps (pre-LN, no positional embeddings, masked to the patch's
   own bytes).
3. **Global latent transformer**: the main body; runs ONCE PER PATCH.
   This is the part that can be SHALLOWER than the current 12 layers.
4. **Local decoder** (§3.3): byte queries ← patch key/values, alternating
   cross-attention + transformer layers; predicts the next byte.
5. **Multi-token prediction** (2404.19737): k parallel prediction heads
   on the global body — better sample efficiency for small models.
6. **BPE-dropout** (ACL2020): stochastic merges during training encode —
   cheap robustness for the existing tokenizer path.

## The hive connection ("plan like the hive, make it fast")

- The patch stream is a hive: patches are cells, boundaries are skipmarks,
  released patches recycle via the freelist (O(1) — no re-tokenization).
- The entropy threshold is a MORPHABLE parameter: overworked → lower
  threshold → shorter patches → more compute; idle → long patches → cheap.
  The amoeba grows/shrinks compute per token without touching a weight.
- Hash n-gram embeddings = one table lookup (no vocab training).
- Entropy model runs with a 2–8 byte window: O(1) amortized, SIMD-friendly.

## The honest risk

BLT's headline wins are at 1B+ scale. At 35M the byte modules might eat
the savings. The experiment (Task 3.3 in the master plan): train a
patched 6-layer body on finemath-live.tok and compare wall-clock loss
trajectory vs the 12-layer BPE baseline. If it doesn't win, keep BPE as
the default and ship BLT as `wubu_mode = 2`.

## Test oracles

1. Patcher round-trip: bytes → patches → bytes == identity.
2. Entropy model CE < BPE tokenizer's byte-equivalent CE on the corpus.
3. Avg patch size ↑ on repetitive text (compute savings real).
4. End-to-end byte-LM loss on a 1M-byte slice beats the BPE path on
   character-level tasks (spelling, noise).
5. MTP: same-token-budget held-out loss floor reached faster.
6. Benchmark gate: same quality at ≤60% flops, decode latency ≤ current.
