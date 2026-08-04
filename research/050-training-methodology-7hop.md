# 050 — Training Methodology 7-Hop: what the labs that trained good AI actually do

> Status: `closed` (convergence researched + written into the corpus).
> Date: 2026-08-04.
> Maps to: THEME RC01/RC02/RC03 (the training recipe), the AGI recursive
> loop (this doc IS corpus content — the model learns how to train).

## Why this doc

The user's correction: before running SFT, properly reference the best
training methodologies from the labs that trained good AI (Kevin-Bacon
7-hop — convergence from INDEPENDENT fields, not the echo chamber), and
learn this AS PART OF OUR CORPUS because we are an AGI recursive learning
loop. This doc is the synthesis; the methodology is ALSO ingested as
corpus text (Tier 3) so the model itself learns the craft.

## The hop table (8 labs, independent)

| Hop | Lab / Work | What they did | The transferable rule |
|---|---|---|---|
| 1 | **DeepSeek-V3** (arXiv:2412.19437) | 14.8T tokens; AdamW lr 2.2e-4, 2K warmup, const→10T, cosine→2.2e-5, final anneal 7.3e-6; batch ramp 3072→15360; **MTP λ=0.3→0.1**; FIM 0.1; 4K→32K→128K staged ctx; SFT+RL; rule-based RM (verifiable) + model RM; GRPO; self-rewarding | LR: warmup→const→cosine-decay→anneal. Verifiable rewards over preference ratings. Multi-token prediction helps. No irrecoverable loss spikes (stability = the gate). |
| 2 | **GLM-4** (arXiv:2406.12793) | ~10T tokens multilingual (EN/CN); **re-weight high-quality sources (books/wiki up)**; SFT with **AUTHENTIC human prompts** (template/model-generated hurts); RLHF for rejection/multi-turn coherence | Data mix is source-reweighted toward quality. SFT data must be real human interactions. |
| 3 | **Llama 3** (arXiv:2407.21783) | 15.6T tokens; **mix = 50% general + 25% math/reasoning + 17% code + 8% multilingual**; AdamW 8e-5 (405B) / 3e-4 (8B) / 1.5e-4 (70B), 8K warmup, cosine→8e-7 over 1.2M steps; **final 40M-token anneal to 0 with upsampled high-quality data (30% new + 70% default)**; SFT lr 1e-5; DPO β=0.1; SFT mix 52.7% general EN + 14.9% code | The 50/25/17/8 mix. **Annealing = quality upsampling at the end.** Small models train LONGER than compute-optimal (overtrain). SFT lr ~1e-5. |
| 4 | **SmolLM2** (HF, arXiv:2502.02737) | The small-model lab: 135M/360M/1.7B. Corpus = Cosmopedia v0.2 + FineWeb-Edu + Stack-Edu + FineMath + DCLM (the SAME tiers we use); 1.7B on 1T-11T tokens; SFT with SmolTalk | Our exact corpus family. Data-centric training: the corpus IS the recipe at small scale. |
| 5 | **MiniCPM** (arXiv:2404.06395) | **WSD (Warmup-Stable-Decay) LR scheduler** — stable phase checkpoints let you branch into decay/domain-adaptation runs without restarting | WSD > plain cosine for continual/adaptive training (one stable ckpt, many decay branches). |
| 6 | **Muon / Kimi Moonlight** (arXiv:2502.16982) | Muon = orthogonalized momentum via Newton-Schulz 5 iters; **Nesterov momentum 0.95, NS5, (a,b,c)=(3.4445,-4.7750,2.0315)**; 3B/16B MoE on 5.7T tokens; SFT works with Muon too; Muon wd 0 (AdamW handles the rest) | Our optimizer IS this (verified: fp32 NS5 needs per-iteration Frobenius renorm). Muon for 2D, AdamW for 1-D. |
| 7 | **Qwen2.5** (arXiv:2412.15115) | 0.5B-72B; up to 18T tokens; two-phase pretrain (4K ctx then extend); **0.5B beats Gemma2-2.6B on math** — tiny models work if data/hyperparams right; DPO+RL hybrid, 150K pairs | Small models are viable. Staged context. Hybrid post-train. |
| 8 | **DeepSeek-R1** (arXiv:2501.12948) | GRPO; **cold-start SFT (thousands of long-CoT examples) → large-scale RL with verifiable rewards → rejection sampling → SFT → final RL**; accuracy reward: correct=+, wrong=0 | Our exact RLVR plan (research/043). Cold start FIRST. Verifiable rewards. |

## The convergence (what 3+ independent labs agree on)

1. **The data mix is 50/25/17/8-shaped**: general+synthetic ~50%, math/reasoning ~25%, code ~15-17%, multilingual ~8% (Llama 3 numbers; GLM/DeepSeek re-weight quality sources up, books/wiki over raw web). OUR mix (cosmopedia 50-60% + finemath 20-25% + openmath 15-20%) IS this shape at 35M scale. ✓
2. **LR schedule = warmup → const → cosine → ANNEAL-to-zero on upsampled high-quality data** (DeepSeek's final anneal, Llama 3's 40M-token anneal, MiniCPM's WSD decay phase — three labs, same principle). Our recipe has warmup+cosine; **the anneal phase is the missing piece.**
3. **The optimizer is Muon for matrices + AdamW for vectors** (Kimi/Moonlight proved it at 16B; GLM-5 and Kimi K2 use it; we already have the real NS5 implementation). DeepSeek used AdamW but the newer convergence is Muon.
4. **SFT cold-start before any RL** (R1's cold start, GLM's authentic-prompts SFT, Llama 3's SFT-at-1e-5): a few thousand REAL human-style interactions at low LR. Our wubu-sft-pack (19,473 convos) is this. SFT lr ~1e-5 (Llama 3) vs our default 1e-3 — **the SFT run must use a LOW lr.**
5. **RL = GRPO with verifiable rewards** (DeepSeek-V3 + R1, the modern consensus): rule-based rewards where possible, model RM otherwise. Our research/043 plan. At 35M: distillation > RL (R1's own finding).
6. **Multi-token prediction helps** (DeepSeek-V3 λ=0.3, the MTP line of work): a cheap companion objective. Worth adding when the trainer supports it.
7. **Training stability is the gate**: DeepSeek's "no irrecoverable loss spikes, no rollbacks" is the bar — our plateau detector + checkpoint/rollback + wubu_priority ledger are the mechanisms.

## What this CHANGES in our recipe (the deltas)

| Knob | Old (research/041) | New (converged) |
|---|---|---|
| schedule | warmup→cosine | warmup→const→cosine→**anneal-to-0 on upsampled quality data** (or WSD) |
| SFT lr | 1e-3 (same as fine-tune) | **1e-5** (Llama 3) — the SFT run is a DELICATE alignment, not a re-train |
| anneal | — | final ~2-5% of tokens: upsampled math/synthetic (Llama 3: 30% new + 70% default) |
| MTP | not in trainer | add λ=0.3 multi-token head when the trainer supports it |
| batch | fixed | ramp (DeepSeek 3072→15360 pattern) — start small, grow |
| stability | plateau detect | + no-rollback gate (already have checkpoint/rollback) |

## Corpus ingestion (the recursive loop)

This methodology is ALSO corpus content. Tier 3 (methodology) is rendered
from this doc + the lab source extracts into role-tagged conversation
text, so WuBu-35M learns the training craft itself (the AGI that knows
how to train AGIs). Build: `tools/build_methodology_tier.py` →
`/home/wubu/models/corpus/methodology/wubu-methodology.txt/.tok`.
