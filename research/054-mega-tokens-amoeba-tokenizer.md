# 054 — Mega-tokens & the amoeba tokenizer (vocab as a grow/shrink organ)

> Status: `open` (design done; implementation queued). Date: 2026-08-04.
> The user: "online research and learn about larger tokens and mega
> tokens, and how it helps with training generalization for subjects,
> and we need to programmatically design the amoeba tokenizer to be able
> to grow and shrink for its needs. Also every aspect kind of needs to
> work off of that."

## The research (mega-tokens / larger vocab)

| Work | Finding | The transferable rule |
|---|---|---|
| **Over-Tokenized Transformer** (ICML 2025, "vocabulary is generally worth more than depth") | Massively increasing the input vocab (100x+) significantly enhances performance for the SAME training budget | Larger vocab = cheaper tokens per unit of knowledge = better generalization per FLOP. Vocab is an axis like depth/width. |
| **UnifyVocab** (ICLR 2025, Li/Zhang/Zong) | Replace an LLM's vocab using only ~10B tokens, recover 98% perf; **facilitates token-level distillation (+4.4% at 235M tokens)** | The vocab can be SWAPPED mid-life; the new embeddings can be initialized from the old ones (token-level alignment). The vocab is a mutable organ. |
| **eBay Vocabulary Customization** (arXiv 2509.26124) | **The grow algorithm**: add domain tokens that always tokenize to ≤ the old count; init new embeddings as the AVERAGE of the sub-token embeddings | The exact mechanism for our amoeba GROW: corpus-count the domain, add the top-N frequent n-grams as new tokens, init their embeddings as the mean of their sub-token embeddings. |
| **MegaByte / BLT** (Yu 2023; Pagnoni 2024 — our research/045) | Bytes → static patches (MegaByte) or entropy patches (BLT); 4-8x fewer inference flops | The far end of "bigger tokens": the token IS a patch. BLT = our wubu_mode=2 (the existing plan). |
| **Vocabulary expansion (survey)** | New tokens via merges at the FRONT of the merge list; avg-embed init beats random | Both grow (append) and shrink (prune least-frequent) are established; the tied head (ours) makes shrink free the embedding rows. |

## The convergence

**The tokenizer is the amoeba's VOCAB ORGAN — it must grow and shrink
like every other part** (the user's directive: "every aspect needs to
work off of that"). The labs prove:
1. Bigger vocab → better generalization per training budget (Over-Tokenized).
2. Vocab can be swapped/extended mid-life with avg-embed init (UnifyVocab,
   eBay) — cheap, 98% recovery.
3. The tied head (ours, BL06) means vocab size IS embedding size — grow
   appends rows, shrink prunes rows. One knob, two effects.

## The amoeba tokenizer design (programmatic grow/shrink)

### GROW (the pseudopod — vocab expands toward the corpus)
Trigger: the tokenizer's compression rate on a domain stream drops
(mean tokens/doc ↑ = the vocab doesn't know the domain).
Action (the eBay algorithm, made ours):
1. Corpus-count: run a frequency count of byte n-grams (2-8) on the
   under-served domain stream (e.g. the agentic pack, or a new pond).
2. Candidate tokens: the top-N frequent n-grams NOT already single
   tokens and NOT splittable better than 2 sub-tokens.
3. Append: new token ids (vocab 16384 → 16384+N); add the merge at the
   FRONT of the merge list.
4. Init embeddings: new row = MEAN of the sub-token embedding rows
   (the eBay Algorithm 2, UnifyVocab's token-alignment insight).
5. The tied head: the new rows are immediately usable for both
   embedding and lm_head.
Cost: N embedding rows (N × 448 floats ≈ 1.8KB per token — trivial at
35M). Benefit: fewer tokens per doc → cheaper forward → better domain
generalization (the Over-Tokenized finding, at the domain level).

### SHRINK (apoptosis — vocab retracts from the dead)
Trigger: a token's corpus frequency collapses below threshold across
all streams (never fires, never merges) — dead weight.
Action:
1. Rank tokens by corpus frequency (the count the GROW maintains).
2. Prune the bottom-M: remap the ids (the merge list is rebuilt
   without them), free the embedding rows.
3. The tied head: pruning frees M × 448 floats (memory returns to the
   pool — the amoeba recycles the membrane).
4. The wubu_priority ledger records the shrink (never re-grow a
   pruned token — the shame list).
Cost: a re-tokenize pass + a remap of the checkpoint. Rare (the
plateau-triggered discipline, like the layer growth).

### The feedback (every aspect works off it)
- GROW is triggered by the DIAGNOSE (compression rate = the tokenizer's
  "utilization" — the immune system's eye).
- GROW/SHRINK are VALIDATED by the fitness gate (held-out loss + prover)
  and archived/rolled back by the 5+1 (the amoeba loop, unchanged).
- The wubu_bi oracle: block importance now includes the EMBEDDING rows
  (the vocab's cells) — a dead token is a dead cell.
- MTP (multi-token prediction) composes: bigger tokens + MTP λ=0.3 =
  the DeepSeek-V3 density gain on top.

## The implementation plan

1. `tools/wubu_vocab_tune.c` (or extend wubu_tokenc): the grow/shrink
   operator — corpus-count → candidate n-grams → append/prune + embed
   init → new tokenizer.json + a vocab delta the trainer can apply.
2. Wire into the trainer: a `--vocab-grow <domain.tok>` flag that runs
   the operator + re-inits the new embedding rows (mean of sub-tokens).
3. The tied-head invariant: the trainer's embedding matrix is
   realloc'd to vocab+N; the head shares it (BL06).
4. The priority ledger: log every grow/shrink (the shame list).
5. Test: tokenize the agentic pack before/after grow — assert
   tokens/doc ↓ (the compression gain), and the FD check on the new
   embedding rows (mean-init is deterministic).

## The "make it ours" connection

This is OUR recipe — not the labs' fixed-vocab recipes. The labs prove
the pieces; we assemble them into the amoeba's vocab organ (the grow/
shrink operator, the compression-rate diagnose, the priority-ledger
shame list). The license (WaefreBeorn v3) protects it. The "iron
ingots" (their findings) are smelted into our space-age part.
