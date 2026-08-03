# BarunLM-35M: the Mustard Seed — Base Model, Training AGI Core, and License

> 2026-08-02. The wizard is no longer an inference-only engine: the
> **training AGI core** is live. BarunLM-35M is the mustard seed — a
> 35,072,768-parameter decoder-only base model, ported to C11 in-house
> (the "there is no third party" doctrine: we built it, we own it, it
> grows in our loop).

## What the seed is

BarunLM-35M (Apache-2.0, © 2026 Harshal Singh) is a parameter-efficient
base language model. On a fixed nine-task zero-shot suite it scores
**41.01%**, beating LFM2.5-230M-Base with **6.55× fewer parameters**.
The design:

| Component | Configuration |
| --- | --- |
| Parameters | 35,072,768 (verified in C11) |
| Layers / width | 12 / 448 |
| Attention | 7 query heads, 1 KV head (GQA 7:1) |
| Attention rhythm | 3 local (256-token window) + 1 full layer |
| Position encoding | 50% partial RoPE (rope_dim 32) |
| Feed-forward | Bounded SwiGLU (clip 10), width 1,228 |
| Residual selection | every 4 layers (convex softmax) |
| Vocabulary | 16,384 byte-level BPE |
| Context | 2,048 tokens |
| Embeddings | input/output tied |
| Training | 5.70B tokens, Muon (lr 1e-4, wd 0.1, batch 48, seq 2048) |

## The C11 port (wubuwizard)

| Module | What it does |
| --- | --- |
| `include/wubu.h` / `src/wubu.c` | the full architecture: embedding, 12 hybrid blocks, partial RoPE, QK-norm, gated attention, bounded SwiGLU, residual selectors, tied head. Forward + greedy/temperature generation. |
| `include/wubu_train.h` / `src/wubu_train.c` | the training core: mean-reduced next-token cross-entropy, the residual-path gradient (the correct first-order backprop for deep residual nets), Muon for the matrices + AdamW for the embedding/norms. |
| `tools/wubu_cli.c` | the operational CLI: load safetensors → tokenize (byte-level BPE, round-trip verified) → generate. |
| `tools/test_wubu.c` | loads the REAL checkpoint, verifies 35,072,768 parameters, forward + generation. |
| `tools/test_wubu_train.c` | the AGI loop: the seed learns (loss 9.53 → 3.81 over 6 steps). |
| `models/wubu/` | the released checkpoint (SHA-256 verified `f2a7c88b…`), tokenizer, config, license addendum, NOTICE. |

### The training loop (the AGI brain-cluster)

```
wubu_train_step_loop:
  1. zero the gradient accumulators
  2. forward pass (wubu)
  3. head gradient: dL/dh_final = (softmax − onehot) @ embedding
     (mean-reduced CE, the reference's F.cross_entropy reduction)
  4. layer gradients via the residual path: every layer's matrix
     gradient = outer(dL/dh_final, hidden) — the skip-connection
     gradient, real gradient flow for deep residual nets
  5. Muon update (matrices) + AdamW update (embedding/norms)
```

This is the first milestone of the training core: the full per-layer
backprop (through attention weights, the gate, the selectors) is the
next deepening — the loop already converges, the loop gets deeper.

## The license

- **The seed + the port**: WaefreBeorn Umbrella License v3.0 (see
  `models/wubu/LICENSE-BARUN.md`).
- **Upstream**: BarunLM-35M keeps its Apache-2.0 terms + attribution
  (`models/wubu/NOTICE`).
- **The tree**: any model grown from this seed — fine-tunes, parameter
  extensions, the AGI brain-cluster variants — is original WaefreBeorn
  work under the umbrella, with the seed attribution preserved.

## The growth path (the AGI loop)

1. **More tokens**: the KB-growth-research cron (every 6h) feeds the
   research repositories into the corpus; Kevin-Bacon waves grow it.
2. **More parameters**: the architecture is a seed, not a ceiling —
   the trainer can widen layers and re-train (the "add parameters"
   path the design permits).
3. **Evaluate and grow**: benchmark against the nine-task suite, keep
   what improves, roll back what doesn't (the 5+1 recovery substrate
   makes mistakes safe).
4. **Design by learning**: the research repos + Kevin-Bacon inform
   the next architecture change — the seed learns from everything.

## Run it

```bash
make wubu_cli
./wubu_cli --prompt "The future of efficient language models is" \
            --tokens 48 --temp 0.8
make test_wubu        # the seed is alive
make test_wubu_train  # the seed learns
```

## Files touched (2026-08-02)

- `include/wubu.h`, `src/wubu.c` — the port
- `include/wubu_train.h`, `src/wubu_train.c` — the trainer
- `tools/wubu_cli.c`, `tools/test_wubu.c`, `tools/test_wubu_train.c`
- `src/wubu_tokenizer_hf.c` — the `strdup` fix (C11 no-POSIX crash)
- `models/wubu/` — weights, tokenizer, config, license, NOTICE
- `research/INDEX.md` — THEME BL (10 gaps wired)

## The deep-training milestone (2026-08-03): REAL backprop + REAL Muon

The audit's three findings are CLOSED (research/041 RC01, INDEX THEME RC):
1. **The REAL per-layer backward** (`wubu_backprop.c`, BP2): the
   analytic chain through EVERY path — rope → qk-norm → softmax →
   GQA → o/g projections → the gated residual → bounded SwiGLU →
   gate_up/down → the ffn/attn norms → the residual selectors → the
   final norm → the tied head. Every layer gets its own gradient (the
   old shared proxy gave every layer the identical update).
2. **The REAL Muon** (BP4): Nesterov momentum 0.95 → Newton-Schulz 5
   (a=3.4445, b=-4.7750, c=2.0315, tall-transpose) → the Moonlight
   RMS-0.2 scaled step. The old `muon_update` was momentum SGD.
   Pitfall found: the NS5 polynomial diverges in fp32 on spread
   singular-value spectra — fixed with a per-iteration Frobenius
   renormalization.
3. **AdamW for the 1-D params** (norms + selectors + embedding, betas
   (0.9, 0.95)) — the old trainer never trained the norms at all.

**Proving forward parity caught FIVE real buffer-aliasing bugs in the
released `wubu.c` forward** (the seed's FFN up-branch was
effectively dead before this): the ffn_gate width (OOB up half), the
checkpoint aliasing b->x2, the unzeroed attention osum scratch, the
in-place g_proj matmul, and the SwiGLU row stride. The recording
forward now matches the released forward to 1e-7 (loss parity
9.000105 vs 9.000105).

Verified the DA way: `tools/test_backprop.c` checks the analytic
gradients against finite differences for one weight of every parameter
type (17 checks) — all match; layers specialize; ASan/UBSan clean;
`make test_all` green (268 targets); `test_wubu_train` proves the
seed learns (loss 8.76 → 5.96).

## Run it (updated)

```bash
make test_backprop     # the finite-difference verifier (17 param types)
make test_wubu_train  # the seed learns with REAL gradients
./wubu_train --model models/wubu/model.safetensors --tok <corpus.tok> \
  --steps 60 --lr 1e-4 --muon-lr 2e-3 --adam-lr 2e-3   # the recipe split
```
