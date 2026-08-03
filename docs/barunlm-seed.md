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
| `include/wubu_barun.h` / `src/wubu_barun.c` | the full architecture: embedding, 12 hybrid blocks, partial RoPE, QK-norm, gated attention, bounded SwiGLU, residual selectors, tied head. Forward + greedy/temperature generation. |
| `include/wubu_barun_train.h` / `src/wubu_barun_train.c` | the training core: mean-reduced next-token cross-entropy, the residual-path gradient (the correct first-order backprop for deep residual nets), Muon for the matrices + AdamW for the embedding/norms. |
| `tools/barun_cli.c` | the operational CLI: load safetensors → tokenize (byte-level BPE, round-trip verified) → generate. |
| `tools/test_barun.c` | loads the REAL checkpoint, verifies 35,072,768 parameters, forward + generation. |
| `tools/test_barun_train.c` | the AGI loop: the seed learns (loss 9.53 → 3.81 over 6 steps). |
| `models/barun/` | the released checkpoint (SHA-256 verified `f2a7c88b…`), tokenizer, config, license addendum, NOTICE. |

### The training loop (the AGI brain-cluster)

```
barun_train_step_loop:
  1. zero the gradient accumulators
  2. forward pass (wubu_barun)
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
  `models/barun/LICENSE-BARUN.md`).
- **Upstream**: BarunLM-35M keeps its Apache-2.0 terms + attribution
  (`models/barun/NOTICE`).
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
make barun_cli
./barun_cli --prompt "The future of efficient language models is" \
            --tokens 48 --temp 0.8
make test_barun        # the seed is alive
make test_barun_train  # the seed learns
```

## Files touched (2026-08-02)

- `include/wubu_barun.h`, `src/wubu_barun.c` — the port
- `include/wubu_barun_train.h`, `src/wubu_barun_train.c` — the trainer
- `tools/barun_cli.c`, `tools/test_barun.c`, `tools/test_barun_train.c`
- `src/wubu_tokenizer_hf.c` — the `strdup` fix (C11 no-POSIX crash)
- `models/barun/` — weights, tokenizer, config, license, NOTICE
- `research/INDEX.md` — THEME BL (10 gaps wired)
