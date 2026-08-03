# RLHF / RLVR Implementation Report for WuBu-35M

**Date:** 2026-08-03 · **Scope:** GRPO-based RLVR (rule-based RL with verifiable rewards) on a 35M-param C11 LLM, using NVIDIA NIM + OpenRouter free-tier APIs as judge oracles. **Bottom line:** implement GRPO (no critic), reward = deterministic verifier (math/code) + binarized LLM-judge vote, SFT cold-start first, β≈0.01, G=8, ε=0.2, lr 1e-6, and treat every continuous judge signal as gameable.

---

## 1. The algorithm we should implement: GRPO (exactly)

**Source of truth:** DeepSeekMath (GRPO paper, [arXiv:2402.03300](https://arxiv.org/abs/2402.03300), §4.1) and DeepSeek-R1 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948), §2). GRPO drops the PPO critic entirely; the baseline is the *group mean reward*.

### 1.1 Objective (per rollout batch)

For each question `q`, sample a group of G outputs `{o_1..o_G} ~ π_θ_old(·|q)`, score them `{r_1..r_G}`, then maximize:

```
J_GRPO(θ) = E[ (1/G) Σ_i (1/|o_i|) Σ_t {
    min[ ratio_t · Â_i , clip(ratio_t, 1−ε, 1+ε) · Â_i ]
    − β · D_KL[ π_θ ‖ π_ref ](o_t)  } ]
```

- `ratio_t = π_θ(o_t|q,o_<t) / π_θ_old(o_t|q,o_<t)` (importance ratio; must recompute `π_θ` log-probs on the *stored* rollouts during the update, and cache `π_θ_old` log-probs at generation time).
- `ε = 0.2` (PPO-standard; TRL default; DeepSeekMath/R1 both use the clipped surrogate).
- KL term is **in the loss, not added to the reward** (this is the key GRPO design choice — it keeps advantage computation clean). Per-token unbiased estimator (Schulman 2020): `D_KL = π_ref/π_θ − log(π_ref/π_θ) − 1` — always positive, needs both ref and current policy log-probs per token.

### 1.2 Advantage (outcome supervision, what R1 uses)

```
Â_i,t = r̃_i = (r_i − mean({r_1..r_G})) / std({r_1..r_G})     for ALL tokens t in o_i
```

Guard: add 1e-4 to std; if a group is degenerate (all rewards equal → std=0), zero the advantages for that group rather than dividing by ~0.

R1's exact hyperparameters ([R1 §2](https://arxiv.org/abs/2501.12948)): G=16, lr=3e-6, β (KL coeff)=0.001, rollout temp=1.0, batch = 32 questions × 16 = 512 completions/step, ref model snapshot refreshed every 400 steps, rollouts (8192) split into 16 minibatches, single inner epoch. DeepSeekMath: G=64, lr=1e-6, β=0.04, max_len=1024, batch=1024, **one policy update per exploration stage** (μ=1).

### 1.3 Recommended settings for WuBu-35M

| Param | Value | Why |
|---|---|---|
| G (group size) | **8** (range 4–16) | TRL default is 8; R1 used 16, DeepSeekMath 64. 35M rollout cost is real; G≥4 needed for a meaningful std. |
| ε (clip) | **0.2** | Standard, works across scales. |
| β (KL coeff) | **0.01** (start; tune 0.001–0.04) | R1 used 0.001 (max exploration), DeepSeekMath 0.04. At 35M we need exploration but also stability → 0.01, add target-KL early-stop. |
| lr | **1e-6** (AdamW, no warmup needed) | DeepSeekMath & TRL default; 3e-6 (R1) is riskier at 35M. |
| μ (inner updates per rollout batch) | **1** (one update; re-sample rollouts each step) | DeepSeekMath "single update following each exploration stage" — simplest correct loop. |
| max_completion_length | 256–512 tokens | R1 used 32k; 35M cannot sustain long CoT. Short CoT only. |
| rollout temp | 1.0, top-p 0.95 | Exploration; R1 uses temp=1. |
| ref refresh | every 200–400 steps | R1: every 400. |
| update batch | 32–64 prompts × G = 256–512 completions/step | R1-scale-shifted down to our compute. |

**C11 engine note:** the trainer needs (a) cached per-token log-probs from generation (`π_θ_old`), (b) a forward pass of the frozen ref model over each stored completion for the KL term, (c) a second forward pass of the *updated* weights for the ratio. GRPO is "off-policy" in the sense that rollouts come from `π_θ_old` — standard autoregressive forward + per-token log-prob extraction suffices; no critic network, no GAE, no value function.

---

## 2. Verifiable rewards at 35M scale (what's gameable and what isn't)

**Do we need a separate reward model?** A trained RM at 35M is *not* viable: an RM must be at least as capable as the policy it scores (Chipper Huyen's RLHF notes, [huyenchip.com/2023/05/02/rlhf.html](https://huyenchip.com/2023/05/02/rlhf.html); R1 explicitly abandoned neural RMs for reasoning because they're "susceptible to reward hacking during large-scale RL", [R1 §2.2](https://arxiv.org/abs/2501.12948)). Instead use the R1 design: `Reward = Reward_acc + Reward_form`, all rule-based ([R1 Eq. 4](https://arxiv.org/abs/2501.12948)).

### Tier 1 — deterministic verifiers (ungameable; the core signal)
- **Math answer matching:** extract final answer (after `answer` tag or `\boxed{}`), normalize (strip units, `$`, whitespace, LaTeX, leading zeros; use sympy equality where available), exact match with ground truth → r ∈ {0,1}. GSM8K (7.5k train) + MATH train (7.5k) both have exact answers.
- **Arithmetic/computation:** Countdown-style target problems and arithmetic chains — evaluate programmatically. (TinyZero uses exactly this; [github.com/Jiayi-Pan/TinyZero](https://github.com/Jiayi-Pan/TinyZero).)
- **Code that compiles + passes hidden unit tests:** run in a sandbox (resource limits, no network); *hidden* test cases only; reward = fraction of tests passed ∈ {0, 1/k, …, 1} or binary all-pass.
- **Structured output:** JSON validity / letter-match (multiple choice) — trivial and safe.

### Tier 2 — LLM-judge oracle (NVIDIA NIM + OpenRouter) — convert scores to rewards *without* gaming
- **Rule:** never use the raw 0–100 judge scalar as reward. Binarize per rubric: `reward = 1 iff judge(s) say correct/valid, else 0` (or 0/0.5/1 for a 3-level rubric where each level is a *separate deterministic check*).
- **Vote across ≥3 judges:** NVIDIA `minimaxai/minimax-m3` + 2 OpenRouter `:free` models (e.g. `google/gemma-4-31b-it:free`, `nvidia/nemotron-3-super-120b-a12b:free`), temperature 0, round-robin across the 6 keys (429s are normal on free tier — see the `free-api-inference` skill: `~/.hermes/profiles/mind-palace/skills/inference/free-api-inference`). Majority vote → binary reward.
- **Keep the judge blind:** judge sees only `prompt + draft`, never the model name, never scores from other judges, never the training status. Fixed prompt, temp 0.
- **Embedding similarity (nv-embed-v1, 4096d) as reward: NO.** Cosine-to-gold-answer similarity is trivially gamed (the model learns to parrot reference phrasing). Use embeddings only as a *diagnostic* (holdout monitoring), not as reward.
- Tülu 3 is the precedent for mixing rule-based + LLM-judge rewards in one RLVR run ([arXiv:2411.15124](https://arxiv.org/abs/2411.15124), §6).

### Tier 3 — format reward (use sparingly)
R1 used `<think>/<answer>` tag enforcement (+Reward_form, equal weight). **SimpleRL-Zoo found rigid format rewards significantly penalize exploration and lower the ceiling for base models that already struggle with instruction-following, and induce overthinking** ([arXiv:2503.18892](https://arxiv.org/abs/2503.18892), §3.1). At 35M: either drop format reward entirely or keep it tiny (+0.1) and only require a delimited final answer (`\boxed{}` / `ANSWER:` prefix) so the extractor works.

**35M reality check:** GRPO can only amplify behaviors already in the model's output distribution. If WuBu-35M cannot emit *any* correct answer on a prompt, the group advantage is zero and RL teaches nothing. Design the prompt set so per-prompt correctness is in the 20–80% band; escalate difficulty as the model improves (curriculum; SimpleRL-Zoo: "the difficulty level of the training data must align closely with the base model's intrinsic exploration capabilities, otherwise zero RL will fail").

---

## 3. Cold start: do we need it? — Yes, effectively mandatory at 35M

- **Evidence R1:** R1-Zero did pure RL from the base model (no SFT) at 671B and it *worked* — emergent reasoning, even an "aha moment" — but produced poor readability and language mixing ([R1 §2–3](https://arxiv.org/abs/2501.12948)). R1 then added a **cold start: thousands of long-CoT examples, human-annotated to a natural conversational style** (B.3.2), before stage-1 RL.
- **Evidence TinyZero:** at small scale the base model matters enormously — Qwen2.5-**0.5B base fails to learn reasoning** on Countdown, 1.5B learns, 3B develops sophisticated reasoning ([TinyZero README](https://github.com/Jiayi-Pan/TinyZero)). A 35M seed is far below the 0.5B failure line; raw RL on the seed will almost certainly produce all-zero rewards.
- **Evidence SimpleRL-Zoo:** zero-RL works on base models only when they already follow instructions reasonably (Qwen2.5 base models are unusually good); other base models need format-reward relaxation + matched difficulty. And their cold-start finding: **high-quality CoT SFT before RL accelerates imitation but limits free exploration** — that tradeoff is acceptable for us (we need basic fluency before RL can differentiate anything).

**Recommendation:** 3-stage pipeline:
1. **SFT cold start** — ~5–20k examples on WuBu-35M: Cosmopedia QA + math CoT (GSM8K/MATH train answers rendered as short CoT) + oracle-improved drafts from the existing `rlhf_oracle.py` loop ([free-api-inference skill scripts/rlhf_oracle.py](https://skill_view: free-api-inference)). Goal is not great reasoning — it's *any plausible answer format* so group rewards become nonzero.
2. **GRPO/RLVR stage** on top of the SFT checkpoint (frozen SFT model = π_ref).
3. **Distillation alternative worth noting:** R1's smallest distilled model (R1-Distill-Qwen-**1.5B**, SFT on 800k R1-generated samples, 2–3 epochs, lr 1e-4) beats pure RL at small scale ([R1 §3.3, B.4.3](https://arxiv.org/abs/2501.12948)). For 35M, supervised training on oracle critiques/improved drafts may be a stronger lever than RL; use RL where the verifier is strong, SFT-on-oracle-outputs elsewhere.

---

## 4. Step-by-step RLHF-off loop with our API resources

```
┌─ Phase 0 · Bootstrap (one-time)
│  1. SFT cold start WuBu-35M on ~5–20k examples (Cosmopedia QA + math CoT +
│     oracle-improved drafts from rlhf_oracle.py via NVIDIA minimax-m3).
│  2. Freeze SFT checkpoint → π_ref (also the initial π_θ, π_θ_old).
│  3. Build prompt bank: GSM8K-train ∪ MATH-train subset (verifiable), Countdown-style
│     arithmetic, ~1–2k Cosmopedia open-ended prompts (judge track). Hold out ~1k
│     prompts per track for monitoring (never trained on).
│
├─ Phase 1 · Rollout (off-policy collection; "RLHF-off" = collect, then update)
│  4. Sample 32–64 prompts/batch; for each, sample G=8 completions from π_θ_old
│     (temp=1.0, top-p 0.95, max 256–512 tokens). Cache per-token log-probs
│     (π_θ_old) at generation time.
│
├─ Phase 2 · Reward (deterministic first, oracle second)
│  5. Math track: extract final answer → normalize → exact/sympy match → r∈{0,1}.
│     Code track: sandbox compile + hidden unit tests → r∈{0,1}.
│     Open-ended track: 3-judge vote (NVIDIA minimax-m3 + 2 OpenRouter :free,
│     temp 0, round-robin keys) → binarized r∈{0,1}. Log every judge raw score
│     separately (never as reward).
│  6. Group-normalize: Â_i = (r_i − mean)/std(±1e-4); zero degenerate groups.
│  7. Forward pass of frozen π_ref over all completions → per-token ref log-probs
│     for the KL term.
│
├─ Phase 3 · Policy update (local, on WuBu-35M C11 engine)
│  8. One GRPO update: re-forward π_θ (current weights) → ratio; clipped surrogate
│     (ε=0.2) − β·D_KL(ref‖θ) per token; β=0.01. lr=1e-6, μ=1 (no inner epochs).
│  9. π_θ_old ← π_θ every step; π_ref ← π_θ every 200–400 steps (R1: 400).
│ 10. Every ~50 steps: holdout eval pass@1 & pass@8 (fresh sampling); monitor
│     reward mean, response length, KL(θ‖ref), token entropy. Early-stop on KL
│     explosion or reward saturation with flat holdout.
│
└─ Phase 4 · Judge hygiene (continuous)
   11. Monthly/periodic judge audit: a *different* judge model re-scores a random
       100-sample slice; track agreement. Judge swap on drift. Never include judge
       critiques verbatim in SFT data without manual verification.
```

Cost at 35M: 2048 rollouts/step ≈ 0.5–1M tokens/step locally on GPU; oracle calls only for the open-ended track (and subsampled judge audits on the math track). 6 OpenRouter keys + NVIDIA NIM absorb the judge load; free tier throttles (429) — round-robin, retry with backoff ([free-api-inference skill](https://skill_view: free-api-inference)).

Implementation reference: TRL's `GRPOTrainer` ([huggingface.co/docs/trl/grpo_trainer](https://huggingface.co/docs/trl/grpo_trainer)) is the battle-tested open implementation to crib from; verl ([verl.readthedocs.io](https://verl.readthedocs.io/en/latest/)) is what TinyZero uses.

---

## 5. Anti-reward-hacking pitfalls (with citations)

1. **Proxy-vs-true divergence (classic RLHF failure):** optimizing any imperfect reward monotonically degrades the true metric — formalized as reward-model overoptimization scaling laws ([Gao et al., arXiv:2210.10760](https://arxiv.org/abs/2210.10760)) and reward hacking surveys ([Pan et al., arXiv:2202.03286](https://arxiv.org/abs/2202.03286); [Skalse et al., arXiv:2202.10085](https://arxiv.org/abs/2202.10085); [Lilian Weng's survey](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/); [Krakovna et al., specification gaming](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)). **Mitigations:** KL anchor to π_ref (β>0), refresh ref, holdout eval that is never touched by training, keep rewards as pure functions of the output.
2. **LLM-judge gaming:** models learn judge quirks — length bias, sycophancy, formatting tricks (RLHF models even learn to mislead human raters; [Wen et al., arXiv:2409.12822](https://arxiv.org/abs/2409.12822)). **Mitigations:** binarize judge scores, ≥3-judge majority vote, temp-0 fixed prompts, blind judges, periodic judge-model swap, and *never* let a continuous scalar flow into the loss.
3. **KL collapse / entropy collapse:** policy turns deterministic, KL(θ‖ref) explodes, or reward saturates at 1.0 while holdout is flat — classic sign the model memorized the verifier. **Mitigations:** β≥0.01, adaptive target-KL early stopping (TRL implements `target_kl`), rollout temp=1.0, monitor entropy; if reward mean hits ~0.95+, escalate prompt difficulty (curriculum) instead of continuing.
4. **Group degeneracy:** all-correct/all-wrong groups → std=0 → advantage blowup or zero signal. **Mitigation:** std+ε guard, zero degenerate groups, and keep per-prompt difficulty in the 20–80% band.
5. **Verifier hacks (RLVR-specific):** models learn to pass *visible* checks without doing the work — hardcoded outputs, exception-swallowing code, code that just prints the expected answer. There is now a dedicated testbed ([Countdown-code](https://github.com/opendilab/awesome-RLVR)); RLVR "remains vulnerable to specification gaming: the verifier and its golden answers can be gamed" (RL4LLM review). **Mitigations:** hidden tests only, sandbox with resource limits, multiple test cases, forbid `if input==`-style short-circuiting, prefer outputs derived from computation.
6. **Reward saturation kills learning:** once pass-rate is ~95%, GRPO advantages ≈ 0 → no gradient. **Mitigation:** curriculum (SimpleRL-Zoo's matched-difficulty finding, [arXiv:2503.18892](https://arxiv.org/abs/2503.18892)).
7. **Cold-start absence at small scale:** all-zero rewards → no signal (TinyZero 0.5B failure). **Mitigation:** SFT cold start + easy-first prompts.
8. **"Aha moment" hype:** response-length growth ≠ emergent reasoning; length increases occur without RL and can be meaningless repetition ([OAT-Zero / "There May Not be Aha Moment in R1-Zero-like Training"](https://sail.sea.com/blog/articles/62)); [NeurIPS "Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?"](https://arxiv.org/abs/2506.14245) shows RLVR mostly *extends the reasoning boundary* (pass@k) rather than creating new capacity. Don't interpret longer outputs as success — use holdout accuracy.
9. **Data contamination:** train/eval overlap inflates scores (R1 does 10-gram dedup against eval, [R1 §D.1](https://arxiv.org/abs/2501.12948)). Keep a fresh holdout.
10. **35M capacity floor:** RL cannot invent behaviors outside the model's sampleable distribution; if the 35M never emits anything near-correct, RL is inert. Distillation/SFT on oracle drafts (R1's own conclusion for small models) is often the better tool at this scale.

---

## Key sources
- GRPO algorithm + equations: DeepSeekMath [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) (§4.1–4.2: objective eq. 3, KL estimator eq. 4, outcome/process advantage, hyperparams)
- RLVR + cold start + distillation: DeepSeek-R1 [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) (§2 rewards/GRPO/aha moment, §3.2 cold start, §3.3 distillation, B.3.2 cold-start data, B.4.3 distill hparams)
- Small-scale evidence: TinyZero [github.com/Jiayi-Pan/TinyZero](https://github.com/Jiayi-Pan/TinyZero); SimpleRL-Zoo [arXiv:2503.18892](https://arxiv.org/abs/2503.18892); RLVR surveys [github.com/opendilab/awesome-RLVR](https://github.com/opendilab/awesome-RLVR)
- Karpathy: "RLHF is just barely RL" [x.com/karpathy/status/1821277264996352246](https://x.com/karpathy/status/1821277264996352246); Deep Dive into LLMs (RLHF section @2:48:26) [youtube.com/watch?v=7xTGNNLPyMI](https://www.youtube.com/watch?v=7xTGNNLPyMI)
- Reward hacking: [lilianweng.github.io/posts/2024-11-28-reward-hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/), [arXiv:2202.03286](https://arxiv.org/abs/2202.03286), [arXiv:2202.10085](https://arxiv.org/abs/2202.10085), [arXiv:2409.12822](https://arxiv.org/abs/2409.12822)
- Implementation refs: TRL GRPO trainer [huggingface.co/docs/trl/grpo_trainer](https://huggingface.co/docs/trl/grpo_trainer); verl [verl.readthedocs.io](https://verl.readthedocs.io/en/latest/); Tülu 3 mixed rewards [arXiv:2411.15124](https://arxiv.org/abs/2411.15124); RM-capability note [huyenchip.com/2023/05/02/rlhf.html](https://huyenchip.com/2023/05/02/rlhf.html)
