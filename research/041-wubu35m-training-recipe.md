# WuBu-35M Training Recipe — How Frontier Labs Actually Train Small LLMs (35M–1B)

Research report, compiled Aug 2026. All numbers below are quoted from primary sources (papers, official repos, author blogs) with URLs at the end of each section and in the Source Index.

---

## TL;DR — Recommended WuBu-35M config (synthesized from all sources)

| Knob | Recommended value | Anchor sources |
|---|---|---|
| Tokens | ≥ 2B total (Chinchilla floor for 35M is ~0.7B; every small-model lab overtrains 10–100× that). If corpus < 2B unique, repeat data ≤ 4 epochs | Chinchilla; TinyLlama; smolLM2-135M; Muennighoff |
| Batch | 256–512 seq × 2048 ctx = 0.5M–1M tokens/step | Karpathy nanoGPT (0.49M); TinyLlama/smolLM2 (2M) |
| Optimizer | **Muon** for all 2D hidden matrices: lr **2e-2** (speedrun used 5e-2), momentum **0.95**, **Nesterov**, Newton-Schulz **5 iters**, bf16, wd **0.1**. **AdamW** for embeddings, lm_head, norms, biases: lr **2e-3** (canonical 3e-4; speedrun 8e-3–6e-1 split), betas (0.9, 0.95), wd 0.1 | KellerJordan/muon; modded-nanogpt; Moonlight |
| LR schedule | Linear warmup **2000 steps** (or ~1% of steps) → cosine to **10% of peak**; or WSD with 10–20% decay | TinyLlama; DeepSeek-V3; smolLM2; nanoGPT |
| Grad clip | 1.0 | TinyLlama; DeepSeek-V3; nanoGPT |
| Context | Start 1024 → extend to 2048 (hybrid attention: local window ~512–1024 + global tokens, warm up the window) | DeepSeek-V3 2-phase extension; modded-nanogpt |
| Eval | Val loss every 125–500 steps on held-out data + HellaSwag/ARC-e/GSM8K (noisy at 35M — trust loss curve) | modded-nanogpt; TinyLlama |
| Post-training | SFT 2 epochs (batch 128, lr 2e-5–5e-5), then optionally DPO (2 epochs, lr 1e-6, β 0.5). Skip RLVR at 35M unless distilling from a bigger teacher | smolLM2 paper; Moonlight; DeepSeek-R1 |

---

## 1. The concrete recipe, stage by stage (with real numbers)

### Stage 0 — Corpus prep & tokenizer

- **Filter + dedup before anything else.** SlimPajama kept only ~50% of RedPajama after cleaning/dedup ([TinyLlama paper §2.1](https://arxiv.org/abs/2401.02385)). FineWeb's pipeline (URL filtering, dedup, quality classifier) is the modern standard ([FineWeb paper](https://arxiv.org/abs/2406.17557)).
- **Mix ratio is a per-stage dial, not a constant.** TinyLlama: SlimPajama : StarCoder = **7:3** ([§2.1](https://arxiv.org/abs/2401.02385)). smolLM2 manually rebalanced web/math/code/instruction mixes **at each training stage** based on the previous stage's evals, and created new datasets (FineMath, Stack-Edu, SmolTalk) where existing ones were too small/low-quality ([smolLM2 paper §1, §4](https://arxiv.org/abs/2502.02737)). For a Cosmopedia+finemath+openmath seed: Cosmopedia ~50–60% (general language), finemath ~20–25%, openmath ~15–20% — and **shift weight toward the math corpora in the final cooldown stage** (Moonlight's cooldown used "the highest quality data, focusing on math, code, and reasoning"; smolLM2's final 10% decay stage introduced FineMath-4+ and InfiWebMath-3+).
- **Tokenizer: train BPE on your own corpus.** TinyLlama used Llama 2's 32K vocab ([§2.2](https://arxiv.org/abs/2401.02385)); modded-nanogpt's world-record runs showed that **halving the vocab (keeping bytes/token) saves a big chunk of params in embeddings+head** ([modded-nanogpt README](https://github.com/KellerJordan/modded-nanogpt)). With vocab 16384 × dim 448, embeddings alone = 7.3M params ≈ **21% of a 35M model** — **tie the embedding and lm_head weights** (shared) or that doubles. A 16K vocab is tight for English+code+math; if you see poor byte-per-token, go 24–32K.
- **Sequence packing:** align batch starts with EoS, cap document length, mask cross-document loss (a modded-nanogpt speed record came from exactly this) ([modded-nanogpt README](https://github.com/KellerJordan/modded-nanogpt)).

### Stage 1 — Pretraining

**Tokens.** Chinchilla-compute-optimal ≈ **20 tokens/param** → 0.7B for 35M ([Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)). Every small-model lab *deliberately overtrains* far past this:
- TinyLlama 1.1B: 3T tokens = ~2,700 tok/param (later cut to 2T — better) ([paper](https://arxiv.org/abs/2401.02385))
- smolLM2-135M: 2T tokens = ~14,800 tok/param; 360M: 4T; 1.7B: 11T ([paper §6](https://arxiv.org/abs/2502.02737))
- MiniCPM: WSD-enabled scaling study found a "much higher compute-optimal data-model ratio than Chinchilla Optimal" ([MiniCPM](https://arxiv.org/abs/2404.06395))

Practical rule for WuBu-35M: **train on every token you have; repeat up to 4 epochs with negligible harm** (repetition ≤4 epochs ≈ free, then diminishing returns — [Muennighoff et al. 2023](https://arxiv.org/abs/2305.16264); TinyLlama itself ran 3 epochs; BabyLM showed 100M words is trainable to strong small-model quality — [BabyLM](https://arxiv.org/abs/2301.11796)).

**Batch.** Karpathy's GPT-2 repro: 12 × 1024 × 40 = **491,520 tokens/step** ([nanoGPT config/train_gpt2.py](https://github.com/karpathy/nanoGPT/blob/master/config/train_gpt2.py)); TinyLlama & smolLM2: **2M tokens/step** ([TinyLlama §2.4](https://arxiv.org/abs/2401.02385), [smolLM2 App. A](https://arxiv.org/abs/2502.02737)). DeepSeek-V3's 3072→15360 sequences is MoE-scale, irrelevant here ([V3 §4.2](https://arxiv.org/abs/2412.19437)). For 35M: 0.5M–1M tokens/step is plenty; with a small batch, raise Adam's β2 (Karpathy used β2=0.99 for a 16K-token batch on the baby model — [train_shakespeare_char.py](https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py)).

**Learning rate.**
- AdamW path: GPT-2/124M repro uses **max 6e-4, min 6e-5 (=10%), β1 0.9, β2 0.95, wd 0.1, warmup 2000 iters, cosine over 600K iters** ([nanoGPT train.py](https://github.com/karpathy/nanoGPT/blob/master/train.py)). For a ~35M baby model you can go higher: Karpathy's 10M char model runs **lr 1e-3** ("with baby networks can afford to go a bit higher" — [train_shakespeare_char.py](https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py)). TinyLlama: max 4e-4, min 4e-5, warmup 2000 steps, 2M-token batches ([§2.4](https://arxiv.org/abs/2401.02385)).
- Muon path: see §3. Matrices 2e-2–5e-2, AdamW parts ~1e-3–3e-4.

**Warmup:** 2000 steps is the near-universal default (TinyLlama, DeepSeek-V3, smolLM2, nanoGPT all use it). For a short 35M run (say 4–8K steps total), ~500–1000 steps or ~1% of steps is fine.

**Schedule:** cosine to 10% of peak (nanoGPT; GPT-3), or **WSD** (warmup-stable-decay — [MiniCPM](https://arxiv.org/abs/2404.06395)): smolLM2-1.7B used WSD, 2000-step warmup, lr 5e-4, **10% linear decay to 0**; the 135M/360M used **20% decay, lr 3e-3** ([smolLM2 §6, App. A](https://arxiv.org/abs/2502.02737)). DeepSeek-V3: warmup 2K steps → constant 2.2e-4 until 10T tokens → cosine to 2.2e-5 over 4.3T → final 500B at 2.2e-5 then 7.3e-6 ([V3 §4.2](https://arxiv.org/abs/2412.19437)). Moonlight: warmup 2K steps to 4.2e-4, cosine to 4.2e-5, then a **cooldown stage**: lr up to 1e-4 in 100 steps, linear to 0 over 500B tokens on math/code/reasoning data ([Muon is Scalable, App. D](https://arxiv.org/abs/2502.16982)).

**Context-length curriculum.** DeepSeek-V3 pretrains at 4K, then two long-context phases (32K → 128K) at constant LR 7.3e-6 ([§4.3](https://arxiv.org/abs/2412.19437)). For WuBu-35M: pretrain at 1024, extend to 2048 for the last ~20–30% of tokens (or WSD-decay stage). For BarunLM's hybrid local/global attention: modded-nanogpt uses **long-short sliding-window attention with window-size warmup + YaRN** (Gemma-2-inspired) ([modded-nanogpt README](https://github.com/KellerJordan/modded-nanogpt)) — warm the local window up from small→full rather than starting full.

### Stage 2 — SFT (cheap at 35M; do it)

- smolLM2 SFT: 2 epochs on SmolTalk, global batch 128, lr in the 1e-5-ish regime ([§5](https://arxiv.org/abs/2502.02737)); Moonlight SFT on tulu-3: **lr 5e-5 → 0 linear, 2 epochs**; Qwen2.5-7B SFT with Muon: lr 2e-5 → 2e-6 cosine ([Muon is Scalable §4](https://arxiv.org/abs/2502.16982)). **Muon works for SFT too** — keep the same param split, just drop the LR.
- Use 100K–1M instruction samples; 1–3 epochs; wd 0.1; grad clip 1.0.

### Stage 3 — RLHF / RLVR (mostly *not* worth it at 35M)

- DeepSeek-R1's pipeline (the reference recipe): cold-start SFT (thousands of examples) → **reasoning RL with GRPO** (group of G outputs per prompt, no critic, KL penalty to a ref policy — [GRPO, DeepSeekMath](https://arxiv.org/abs/2403.17031)) → **rejection sampling from the RL checkpoint → second SFT (~800K samples)** → full-scenario RL ([DeepSeek-R1 §3](https://arxiv.org/abs/2501.12948)). R1 evaluates with temperature 0.6, top-p 0.95, 4–64 samples ([§4.1](https://arxiv.org/abs/2501.12948)).
- **Key small-model finding from R1: distillation beats RL.** "Direct distillation from DeepSeek-R1 outperforms applying RL on it" for the ≤32B Qwen/Llama models; R1 distilled 800K samples into 1.5B–70B models ([§1.1](https://arxiv.org/abs/2501.12948)). For a 35M seed: skip RLVR, or distill from a larger model; if you must RL, use nanochat's harness (Karpathy's current all-stages repo, incl. RL) ([nanochat](https://github.com/karpathy/nanochat)).
- DPO alternative at small scale: smolLM2-instruct used **DPO, 2 epochs, lr 1e-6, β 0.5, global batch 128, seq 1024** ([§5](https://arxiv.org/abs/2502.02737)).

---

## 2. What specifically matters at 35M vs 1B+

1. **Optimizer & LR dominate.** Muon's biggest wins are at small scale — it took the GPT-2-124M speedrun from 31.4 → 24.9 min the day it was introduced ([modded record history](https://github.com/KellerJordan/modded-nanogpt)), and Moonlight's scaling-law study measured **~2× compute efficiency (52% of AdamW FLOPs)** ([Muon is Scalable](https://arxiv.org/abs/2502.16982)). At 35M, the LR/optimizer choice moves results more than at 1B+.
2. **Data > params.** 35M models are under-trained by default; the whole point of TinyLlama/smolLM2/MiniCPM is *overtraining* small models. Feed all tokens, repeat ≤4 epochs, and finish on a high-quality cooldown.
3. **Don't copy big-model batch sizes.** 2M+ token batches (TinyLlama/smolLM2) and 63M (DeepSeek-V3) are for throughput at scale. 0.5M tokens is the GPT-2/124M anchor; at 35M you can go 0.25M–1M. Smaller batch → use β2 0.99 or Muon.
4. **Embedding/head share of the param budget is huge.** 16384×448 = 7.3M ≈ 21% of 35M; untied head doubles that. Tie embed+head; train the tokenizer on your corpus; consider vocab 24–32K if 16K under-tokenizes.
5. **Eval noise.** Benchmarks at 35M are near-chance and noisy; track held-out loss as the primary health metric, evaluate every 125–500 steps, and don't tune on single benchmark seeds ([modded-nanogpt val_loss_every=125](https://github.com/KellerJordan/modded-nanogpt)).
6. **Repetition is expected, but order matters.** TinyLlama *regressed* from 3T → 2T tokens (fewer epochs better) ([§2.5](https://arxiv.org/abs/2401.02385)); repeated data must be reshuffled across epochs, and the final cooldown should upsample math/code ([Moonlight App. D](https://arxiv.org/abs/2502.16982); [smolLM2 §4](https://arxiv.org/abs/2502.02737)).
7. **Sanity check the param count.** 448-dim × 32 layers with FFN hidden 2048 is ~3.6M params/layer ≈ 115M + 7.3M embedding ≈ **~120M, not 35M** — unless embeddings are shared and layers/width differ. Verify before locking the recipe; if it truly is 35M, the architecture is smaller than described (e.g. ~9 layers), which is fine — everything above still applies.

---

## 3. Muon — correct usage, and confirm/correct for the Barun trainer

### The algorithm (authoritative)

Muon = **MomentUm Orthogonalized by Newton-Schulz** ([Keller Jordan, blog post](https://kellerjordan.github.io/posts/muon/), [repo](https://github.com/KellerJordan/Muon)):

```
m_t = μ·m_{t-1} + g                          # momentum (μ = 0.95), fp32 buffer
O_t = NewtonSchulz5(m_t)                     # orthogonalize (or Nesterov variant: NS(g + μ·m))
W_t = W_{t-1} − η·O_t                        # no bias correction, no per-param sqrt scaling
```

- **Newton-Schulz 5 iterations** (5 = "sweet spot"; 10 is more accurate but no better — [Moonlight §2.2](https://arxiv.org/abs/2502.16982)):
  coefficients **(a,b,c) = (3.4445, −4.7750, 2.0315)**; normalize `X = G/‖G‖_F`; **transpose if rows > cols**; run the iteration in **bf16** (it's numerically stable there, unlike coupled-Newton) with eps 1e-7 ([blog](https://kellerjordan.github.io/posts/muon/); original speedrun code `newtonschulz5`).
- **Momentum 0.95**, Nesterov-style (`nesterov=True` in the speedrun; Moonlight also applies NS to `μ·m + g`).
- **No bias correction** (unlike Adam).
- **Param split (the part Barun already implements — confirmed correct):** 2D matrices of *hidden* layers (attention q/k/v/o, MLP in/out/gate) → Muon. **Embeddings, lm_head, RMSNorm weights, biases, all 1-D params → AdamW.** Both Keller ("scalar and vector parameters… as well as the input and output layers, should be optimized by a standard method such as AdamW" — [blog](https://kellerjordan.github.io/posts/muon/)) and Moonlight ("AdamW is used in couple with Muon to handle non-matrix based parameters, like RMSNorm, LM head, and embedding parameters" — [§2.2](https://arxiv.org/abs/2502.16982)) are explicit. Muon on embed/head **hurts** — this is the most common Muon bug.
- **LRs:** canonical example from the Muon repo: Muon group **lr 0.02**, AdamW group **lr 3e-4** with betas (0.90, 0.95), wd 0.01 both ([Muon README](https://github.com/KellerJordan/Muon)). The original speedrun (124M, 3.28 FineWeb loss): **Muon lr 0.05**, and three separate Adam groups with betas (0.8, 0.95): embeddings 0.6, lm_head 0.008, scalars 0.04 ([train_gpt2.py @ 9730304](https://github.com/KellerJordan/modded-nanogpt/blob/973030408364f8738b4ad9e8f912d8cbbf56e4d4/train_gpt2.py)) — those are speedrun-tuned, not defaults.
- **Weight decay:** the canonical example uses wd 0.01; Moonlight found **wd 0.1 is essential at scale** — without it, weight and layer-output RMS grow beyond bf16 range (their fix #1) ([§2.2](https://arxiv.org/abs/2502.16982)). The original speedrun used wd 0 and added "standard weight decay" later ([record history](https://github.com/KellerJordan/modded-nanogpt)). **Recommendation for 35M: wd 0.1 on both groups** (0.01–0.1 is the safe band; some labs zero wd on embeddings).
- **Per-matrix update scale (Moonlight fix #2, optional at 35M):** either normalize each NS update so its RMS = 0.2, or scale LR per matrix by `0.2·√max(A,B)`. This lets Muon reuse AdamW-tuned LR/wd and prevents instability on tiny/odd-shaped matrices (e.g. individual GQA KV heads) ([§2.2](https://arxiv.org/abs/2502.16982)). If BarunLM's 8Q/1KV GQA or hybrid-attention projections misbehave, apply this.
- **Grad clip 1.0; momentum buffer in fp32** (the speedrun comment: "FP32 for precision").

### Confirm/correct checklist for the Barun trainer

| Item | Correct practice | Status to check |
|---|---|---|
| Matrix split | hidden 2D → Muon; embed/head/norm/bias → AdamW | ✅ you have this — keep it |
| NS iterations | 5 (not 3, not 10) | verify |
| NS coefficients | (3.4445, −4.7750, 2.0315) | verify |
| NS normalization | divide by ‖G‖_F; transpose tall matrices; bf16 | verify |
| Momentum | 0.95 (not 0.9), Nesterov variant OK, fp32 buffer | verify |
| Bias correction | none (do not port Adam's bias-corr into Muon) | verify |
| LR ratio | Muon ≈ 10–60× the AdamW-group LR (0.02–0.05 vs 3e-4–2e-3) | verify |
| Weight decay | 0.1 on matrices (and 0.1 or 0.0 on embed/head); wd 0 on Muon = RMS growth at scale | verify |
| Grad clip | 1.0 | verify |
| Embed/head | must NOT be Muon-optimized | verify |

---

## 4. Top-10 gotchas that make a 35M seed fail to learn

1. **Wrong LR for the optimizer.** AdamW's 6e-4 applied to Muon layers ≈ no learning; Muon needs ~1e-2–5e-2. Conversely 6e-4-scale AdamW on embeddings is too high. ([Muon blog](https://kellerjordan.github.io/posts/muon/); [Muon repo](https://github.com/KellerJordan/Muon))
2. **No warmup + high LR → loss spike / divergence.** Every reference uses linear warmup (~2000 steps or ~1% of total): TinyLlama, DeepSeek-V3, smolLM2, nanoGPT. ([TinyLlama §2.4](https://arxiv.org/abs/2401.02385), [nanoGPT train.py](https://github.com/karpathy/nanoGPT/blob/master/train.py))
3. **Batch too small (<~128K tokens) with AdamW → noisy, unstable at any decent LR.** Karpathy's baby model compensates with β2=0.99; better: 0.5M-token batches. ([train_shakespeare_char.py](https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py), [train_gpt2.py](https://github.com/karpathy/nanoGPT/blob/master/config/train_gpt2.py))
4. **Data repetition/order.** TinyLlama v1.1 *got worse* with 3T (3 epochs) vs 2T (2 epochs); repeating in the same order causes memorization spikes. Repeat ≤4 epochs, reshuffle each epoch, cooldown on the best data. ([TinyLlama §2.5](https://arxiv.org/abs/2401.02385), [Muennighoff](https://arxiv.org/abs/2305.16264), [Moonlight App. D](https://arxiv.org/abs/2502.16982))
5. **Muon weight-decay sins.** No wd at all → RMS growth → bf16 overflow (Moonlight's #1 finding); wd applied to embeddings/head when the rest is clean — either is a silent quality killer. ([Muon is Scalable §2.2](https://arxiv.org/abs/2502.16982))
6. **Muon on embeddings / lm_head / norms.** Must be AdamW; this is the #1 Muon mis-application and it measurably hurts. ([blog](https://kellerjordan.github.io/posts/muon/), [§2.2](https://arxiv.org/abs/2502.16982))
7. **Newton-Schulz implementation bugs:** <5 iterations, missing Frobenius normalization, no tall-matrix transpose, fp32 (slow) or fp16 (unstable) instead of bf16, wrong coefficients. All spelled out in the reference code. ([newtonschulz5](https://github.com/KellerJordan/modded-nanogpt/blob/973030408364f8738b4ad9e8f912d8cbbf56e4d4/train_gpt2.py))
8. **Tokenizer/corpus mismatch.** A 16K vocab BPE trained on the wrong corpus wastes ~21% of the model on embeddings and caps quality; verify bytes/token on the actual corpus; tie embed+head. ([modded-nanogpt README](https://github.com/KellerJordan/modded-nanogpt))
9. **No eval discipline.** Evaluate every 125–500 steps on held-out data; at 35M benchmarks are near-chance noise — trust the val-loss curve, watch the train/val gap when repeating data. ([modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt))
10. **Scheduler/cooldown mistakes.** Cosine to 0 with no min (use 10% floor — nanoGPT); or WSD without a decay stage at all (the decay/cooldown is where the model actually converges — smolLM2 used 10–20% of tokens for it; TinyLlama substitutes a 4× batch-size cooldown). Plus: **hybrid-attention misconfig** — BarunLM's local/global attention needs the local window warmed up and loss on global tokens; skip this and long-range/global capacity silently dies. ([nanoGPT train.py](https://github.com/karpathy/nanoGPT/blob/master/train.py), [smolLM2 App. A](https://arxiv.org/abs/2502.02737), [TinyLlama §2.4](https://arxiv.org/abs/2401.02385), [modded-nanogpt README](https://github.com/KellerJordan/modded-nanogpt))

---

## Source index (all URLs cited)

- Keller Jordan, "Muon: An optimizer for hidden layers in neural networks" (blog) — https://kellerjordan.github.io/posts/muon/
- KellerJordan/muon (canonical usage + hyperparams) — https://github.com/KellerJordan/Muon
- modded-nanogpt (NanoGPT speedrun; record history; original Muon config @ commit 9730304) — https://github.com/KellerJordan/modded-nanogpt
- Liu et al. (Moonshot/Kimi), "Muon is Scalable for LLM Training" (Moonlight) — https://arxiv.org/abs/2502.16982
- Zhang et al., "TinyLlama: An Open-Source Small Language Model" — https://arxiv.org/abs/2401.02385
- Ben Allal et al. (HuggingFace), "SmolLM2: When Smol Goes Big" — https://arxiv.org/abs/2502.02737
- DeepSeek-AI, "DeepSeek-V3 Technical Report" — https://arxiv.org/abs/2412.19437
- DeepSeek-AI, "DeepSeek-R1" — https://arxiv.org/abs/2501.12948
- Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning… (GRPO)" — https://arxiv.org/abs/2403.17031
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla) — https://arxiv.org/abs/2203.15556
- Muennighoff et al., "Scaling Data-Constrained Language Models" — https://arxiv.org/abs/2305.16264
- Hu et al., "MiniCPM… (WSD scheduler)" — https://arxiv.org/abs/2404.06395
- Warstadt et al., "BabyLM Challenge" — https://arxiv.org/abs/2301.11796
- Karpathy, nanoGPT (train.py defaults: lr 6e-4, warmup 2000, min_lr 6e-5, β2 0.95, clip 1.0, wd 0.1) — https://github.com/karpathy/nanoGPT
- Karpathy, nanochat (all-stages small-model harness, auto-scaled hyperparams, RL) — https://github.com/karpathy/nanochat
- Karpathy, LLM101n (course) — https://github.com/karpathy/llm101n ; "Let's build GPT" — https://www.youtube.com/watch?v=kCc8FmEb1nY
- Penedo et al., "The FineWeb Datasets" — https://arxiv.org/abs/2406.17557
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2) — https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
