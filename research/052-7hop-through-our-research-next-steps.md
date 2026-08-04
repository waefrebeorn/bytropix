# 052 — The 7-hop through OUR research: what to do right now

> Status: `closed` (converged + executed). Date: 2026-08-04.
> The user: "do a seven steps research through our research to find what
> we should be doing right now because we have a library of information
> now." This is the 7-hop over OUR OWN docs (not the web — the library
> IS the source now).

## The 7 hops (all from our own research)

| Hop | Source (ours) | What it says | The rule |
|---|---|---|---|
| 1 | research/050 (lab methodology 7-hop) | Muon+AdamW split, WSD/anneal, SFT cold-start at LOW lr (~1e-5), GRPO later | SFT lr is DELICATE — 1e-5, not 1e-3 |
| 2 | research/051 (Triple-DA) | the lab recipe is the amoeba's METABOLISM; biggest risk = under-training | train on the local corpus NOW, don't waste oracle tokens |
| 3 | research/041 (training recipe) | Stage 2 = SFT 2 epochs, lr 2e-5–5e-5, batch 128; skip RLVR at 35M | the SFT run is the current stage |
| 4 | research/042 (data curation) | mix 70% cosmopedia / 22% finemath / 8% openmath; annealing up-samples math | our Tier-0 mix matches; chat templates are the SFT format |
| 5 | research/048 (SFT corpus) | wubu-sft-pack.jsonl: 19,473 conversations (GSM8K math CoT + UltraChat + SlimOrca) | tokenized: wubu-sft.tok (12M tokens) — READY |
| 6 | research/043 (RLHF) | SFT cold-start FIRST, freeze as π_ref, then GRPO; distillation > RL at 35M | the live NVIDIA oracle comes AFTER the SFT run |
| 7 | THEORY/01 + wubu-amoeba-design + research/051 | geometry IS architecture; the body learns via the metabolism; under-trained = the biggest risk | get the BASE loss down first; the amoeba grows after |

## The convergence (what we should do RIGHT NOW)

1. **Burn NO NVIDIA tokens on feedback** — the base model at ~7.7 loss
   produces degenerate drafts (verified: token-205 repetition). The
   oracle is for AFTER the model can say something. (research/043:
   SFT cold-start first.)
2. **Run the SFT cold-start on the LOCAL corpus** — wubu-sft.tok (12M
   tokens, 19,473 chat-template conversations with <system>/<user>/
   <assistant> markers IN the vocab). This is the chat-template focus
   the user asked for: the model learns the conversation format.
3. **Use the converged SFT lr ~1e-5** (research/050 hop 1 + 041 Stage 2:
   lr 2e-5–5e-5) — NOT the fine-tune 1e-3. Low LR = the alignment is
   delicate.
4. **Train, watch loss DOWN** — the goal is the held-out loss falling
   (research/041: "trust loss, not benchmarks at 35M"). The user's
   "get our scores down" = loss down on the chat-template corpus.
5. **Checkpoint + resume** — save as seed-sft.st; the 5+1 recovery and
   wubu_priority ledger guard it (the amoeba's safety).
6. **After SFT: freeze π_ref, THEN the live NVIDIA oracle + GRPO.**
7. **The amoeba grows after the base can talk** — the corpus waves
   (ponds, agentic, methodology) are the food; the body grows when the
   metabolism works.

## The execution plan (this session)

1. Fix the trainer's 4M-token corpus cap → 12M (wubu-sft.tok fits).
2. Run wubu_train_cli with --tok wubu-sft.tok --lr 1e-5 (SFT mode),
   resume from seed-48.st (the 500-step pretrained base).
3. Verify loss DOWN on the chat-template corpus.
4. Checkpoint seed-sft.st; commit; update the slate.

## Why NOT the live oracle yet (the honest DA)

The live loop is BUILT (wubu_live_learn + nvidia_nim.score_draft, all
verified live). But the model at loss ~7.7 generates token-205
repetition — scoring that wastes NVIDIA credits on noise. The lab
convergence (R1 cold start, SimpleRL-Zoo format-reward finding) and our
own research/043 all say: SFT cold-start FIRST, oracle AFTER. Burn the
credits when they teach something.
