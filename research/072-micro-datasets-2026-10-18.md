# 072 — Micro-dataset pack (2026-10-18)

The ask: research a spread of *small, genuinely good* Hugging Face datasets
under the 25 / 7 / 25 regimen, then actually obtain them.

## Method

25 queries × 7 hops × 25 on top, applied to the HF catalogue instead of arXiv
(because the artifact wanted is a dataset, not a paper):

| Hop | Question | Tool |
|---|---|---|
| 1 | What exists per topic? | HF `/api/datasets?search=` × 36 topic queries, sorted by downloads |
| 2 | Which are small? | `size_categories` tag → `tree?recursive=true` for **exact bytes** |
| 3 | Which are real? | ungated check + file count + container magic (`PAR1`, JSON/JSONL) |
| 4 | What is inside? | `datasets-server/first-rows` → actual field names |
| 5 | Which are licensed? | `license:` tag, permissive set only |
| 6 | What did the field miss? | second 25-query pass on OS / terminal / compiler / security axes |
| 7 | What survives? | hand selection, contamination labelling, download |

## Funnel (real counts)

| Stage | Count |
|---|---|
| Discovery queries (36 + 25) | 61 |
| Unique datasets seen | 841 + 344 |
| With a small `size_categories` tag | 332 |
| Exact size measured, ≤ 300 MB | 236 |
| Permissive licence + usage signal | 79 |
| Schema verified via first-rows | 27 |
| Downloaded into the pack | 26 |
| Total bytes | ~1.04 GB |

Search hits are not citations: layers 1–2 are catalogue ranking, not quality
proof. The quality proof here is the schema probe plus the download.

## The pack

`/home/wubu/models/corpus/micro/<owner>__<name>/` with a per-dataset
`MANIFEST.json` and a pack-wide `INDEX.json`.

| Dataset | Cat | Licence | MB | Why it earns a slot |
|---|---|---|---|---|
| agentica-org/DeepScaleR-Preview-Dataset | reasoning | MIT | 21.5 | long-CoT math distilled from a strong teacher; compact |
| KbsdJames/Omni-MATH | math | Apache-2.0 | 7.5 | olympiad-level, difficulty + domain labels |
| reasoning-machines/gsm-hard | math | MIT | 1.1 | hand-written hard GSM with verifiable code |
| DigitalLearningGmbH/MATH-lighteval | math | MIT | 9.7 | clean 7.5k competition set with levels |
| EleutherAI/hendrycks_math | math | MIT | 4.9 | the canonical MATH split, small and stable |
| sahil2801/CodeAlpaca-20k | code | CC-BY-4.0 | 8.1 | instruction→code baseline |
| SWE-bench/SWE-bench_Multilingual | code | MIT | 1.9 | **eval only**; patch-level repo tasks |
| ScalingIntelligence/KernelBench | code | Apache-2.0 | 0.1 | **eval only**; CUDA kernel synthesis prompts |
| Infatoshi/kernelbench-mega-traces | code | MIT | 169.6 | full agent traces solving kernel tasks (tool calls, harness output) |
| code-review-bench/code-review-bench | code | CC-BY-4.0 | 28.0 | PR diffs with human review outcomes |
| NousResearch/hermes-function-calling-v1 | tools | Apache-2.0 | 66.7 | multi-turn tool calls with schemas |
| glaiveai/glaive-function-calling | tools | Apache-2.0 | 113.2 | broad function-calling coverage |
| interstellarninja/tool-use-multiturn-reasoning | tools | Apache-2.0 | 26.6 | reasoning interleaved with tool use |
| nvidia/Aegis-AI-Content-Safety-Dataset-2.0 | safety | CC-BY-4.0 | 28.4 | prompt+response labels, violation taxonomy |
| thu-coai/SafetyBench | safety | MIT | 10.4 | **eval only**; 7 safety categories |
| AlignmentResearch/DolusChat | safety | CC-BY-4.0 | 65.2 | deceptive-behaviour dialogues with feature labels |
| AxiomicLabs/Tiny_Theory_of_Mind | science | Apache-2.0 | 2.7 | theory-of-mind multiple choice, tiny |
| HuggingFaceH4/Multilingual-Thinking | multilingual | Apache-2.0 | 5.3 | reasoning in the user's language |
| CohereLabs/aya_evaluation_suite | multilingual | Apache-2.0 | 22.0 | **eval only**; 100+ languages |
| fka/prompts.chat | instruction | CC0-1.0 | 7.3 | the classic prompt collection, public domain |
| allenai/tulu-3-sft-personas-instruction-following | instruction | ODC-BY | 39.2 | persona-conditioned constraints |
| google/Synthetic-Persona-Chat | instruction | CC-BY-4.0 | 38.4 | two-persona chat synthesis |
| FreedomIntelligence/medical-o1-reasoning-SFT | medicine | Apache-2.0 | 247.5 | clinical CoT with answers |
| gbharti/finance-alpaca | finance | MIT | 42.9 | finance instruction set |
| harborframework/terminal-bench-2.1 | os-agent | Apache-2.0 | 19.4 | **eval only**; terminal tasks with env configs |
| jedisct1/security-audits | os-agent | MIT | 51.1 | real audit sessions with tool traces |

## Contamination rule

Five entries are benchmarks, not training data:

`SWE-bench_Multilingual`, `KernelBench`, `terminal-bench-2.1`, `SafetyBench`,
`aya_evaluation_suite`.

They are listed in `docs/micro-datasets.json → eval_only` and must stay out of
every training mix. A model trained on its own eval set reports fiction.

## What this changes for WuBu

- **OS agent role**: `terminal-bench-2.1` + `security-audits` +
  `kernelbench-mega-traces` give tool-call traces grounded in real shells and
  kernels, not chat. This is the corpus shape WuBuOS needs.
- **Tool calling**: three complementary sets (Hermes, Glaive, multi-turn
  reasoning) cover schema following, breadth, and interleaved reasoning.
- **Safety**: Aegis gives a label taxonomy; DolusChat gives deception
  positives; SafetyBench measures.
- **Small is the point**: 1.15 GB total, so the whole pack tokenizes and
  trains in minutes on the 13 GB box, and can be re-mixed per experiment.

## Next

1. Tokenize the pack with the C11 BPE into `corpus/micro/tokens/`.
2. Dedupe against the existing SFT and reasoning tiers (near-dup, not hash).
3. Build `micro-sft-pack.jsonl` at ~150M tokens with a documented mix,
   keeping the five eval sets out.
4. Measure on the eval sets *before* training, so any later gain is real.
