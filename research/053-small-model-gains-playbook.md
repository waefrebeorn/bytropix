# 053 — Small-model gains: the Smol Training Playbook + lab convergence (7-hop online)

> Status: `open` (synthesis done; the top gains queued for implementation).
> Date: 2026-08-04. The user: "do an online research seven steps of all
> of the small models and all of the best labs... use their elements and
> their iron ingots to create our space age technology."
> Primary source: The Smol Training Playbook (HuggingFaceTB, Oct 2025,
> Loubna Ben Allal / Lewis Tunstall / Nouamane Tazi / Elie Bakouch /
> Ed Beeching / Colin Raffel / Leandro von Werra / Thomas Wolf).
> FULL TEXT ARCHIVED: ~/.hermes/profiles/mind-palace/cache/web/gist.github.com-9b750a1d00.md (599 KB).

## The 7 hops (online, the small-model labs)

| Hop | Source | The transferable gain for WuBu-35M |
|---|---|---|
| 1 | Smol Training Playbook (HF) | **MASK USER TURNS in SFT** — loss on assistant tokens ONLY (a few points, big on IFEval). Our trainer currently trains all tokens incl. the `<user>` prompt. |
| 2 | Smol Training Playbook | **Packing** — we already pack (.tok = concatenated docs) ✓ |
| 3 | OLMo2 (via playbook) | **No weight decay on embeddings** — wd on embed norms lowers them, destabilizes early layers. Our recipe has wd 0.1 on everything. |
| 4 | Llama 3.2 (Meta) | **Distillation recovers pruned small models** — 1B pruned from 8B, KD from 8B/70B logits. We OWN Qwen3.6-27B / KAT-Coder / Agents-A1-4B locally → **teacher logits for WuBu-35M = the "iron ingots"**. |
| 5 | Sheared-LLaMA / SLM survey | **Targeted structured pruning + dynamic batch loading** — prune big → small, adapt data mix by per-domain loss. The amoeba's shrink, done with lab data. |
| 6 | Small-batch SGD (arXiv 2507.07101) | Small batches + momentum converge fine; gradient accumulation wasteful — our seq-2048-per-step is the right shape ✓ |
| 7 | Qwen2.5 (0.5B beats Gemma2-2.6B on math) | Small models are viable with the right data/hyperparams — the under-training risk (051) is the thing to fix, not the architecture |

## The convergence (what to implement, in order)

1. **SFT user-turn masking** (highest leverage for the RUN happening now):
   the loss should only cover assistant tokens. In the trainer, that means
   a per-token loss mask: positions inside `<user>...</user>` are masked
   (loss 0), `<assistant>...</assistant>` and beyond are trained.
   Implementation: detect the `<user>` (id 5) / `<assistant>` (id 6) span
   boundaries in the token window and zero the loss on user spans.
   NOTE: our rendered SFT text is `<user>Q</user>\n<assistant>A</assistant>`
   — masking user spans is a clean, bounded change to
   wubu_train_step_loop's loss computation.
2. **No weight decay on embeddings** — split the wd: Muon matrices keep
   wd 0.1, embeddings get wd 0.0 (OLMo2-validated, free stability).
3. **Distillation from our local teachers** — the big "use their iron
   ingots" item: run Qwen3.6-27B (or KAT-Coder) logits on the SFT/agentic
   corpus, train WuBu-35M to match (KL on teacher logits + CE on truth).
   We have the models locally; the forward pass exists; this is the
   Llama-3.2 recipe at 35M scale.
4. **Loss-spike survival** (playbook: "training is war") — we already have
   the 5+1 recovery + plateau detector; the playbook validates the
   fix-and-restart discipline.

## What this does NOT change

- The chat template (ChatML-style `<system>/<user>/<assistant>` — the
  playbook says ChatML is the right default; our tokenizer ids 4-6 match).
- Muon+AdamW split (the playbook's "AdamW and beyond" section validates
  the optimizer-family direction; Moonlight's Muon is our edge).
- The SFT-first pipeline (playbook: "almost every post-training pipeline
  starts with SFT — it's cheap, stable, the right baseline").
- WSD/anneal (playbook's schedule section covers WSD; we chose it).

## The honest DA (what could go wrong)

1. Masking changes the loss scale (fewer trained tokens) — the lr may
   need a bump; measure loss-delta, not absolute loss.
2. Distillation needs the teacher's logits over OUR vocab (16,384) — the
   teacher (Qwen 152K vocab) must map tokens; the embedding mismatch is
   the real work. Alternative: distillation on the HIDDEN states after a
   learned projection (FitNet-style) — cheaper, vocab-agnostic.
3. No-wd-embeddings interacts with tied embeddings (our embed == head) —
   the head is 2D, Muon'd; keep the tie, drop wd on the shared matrix.

## Status

- ✅ Playbook archived (persistent source).
- 🔄 SFT cold-start run in progress (research/052: 12M tokens, lr 1e-5,
  seq 2048; checkpoints landing).
- ⏳ Next: implement user-turn masking in wubu_train_step_loop; then the
  no-wd-embeddings split; then distillation from local teachers.

## References

- The Smol Training Playbook — archived full text (cache/web/...9b750a1d00.md)
- Llama 3.2 (ai.meta.com) — pruning + KD recovery for 1B/3B
- OLMo2 (arXiv 2501.00656) — no-wd-embeddings stability
- Sheared-LLaMA — structured pruning + dynamic batch loading
- Small Batch Training (arXiv 2507.07101) — SGD/momentum with small batches
- Qwen2.5 (arXiv 2412.15115) — small-model viability
