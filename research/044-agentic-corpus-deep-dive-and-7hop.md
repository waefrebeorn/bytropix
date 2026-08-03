# The AGI-user corpus deep-dive and 7-hop chain (2026-08-04)

**Target:** the best "AGI-user corpus" — expert trajectories of human-AGI
interaction (tool use, reasoning traces, computer use, long-horizon tasks) —
for a USABLE agentic LLM. The standing loop's corpus side: this wave tells
us what the *agentic* corpus mix must be, on top of the 4.44B pretraining
tokens already streamed (FineMath + OpenMath, 2026-08-03).

## The corpus landscape (measured 2026-08-04)

| Corpus / system | Scale | The mechanism that matters |
|---|---|---|
| **Hermes 4** (Nous, arXiv 2508.18255) | 5.1M samples / 19B tokens (3.5M reasoning + 1.6M non-reasoning) | DataForge (graph synthesis, PDDL pre/postconditions, nested DAGs, judge ≠ answer weights, train on intermediate calls) + Atropos RL envs (Answer-Format 150+ formats, RLVR-IFEval, Internbootcamp 70K rejection-sampled trajectories, Schema Adherence w/ dynamic Pydantic, Tool-Use taxonomy DFS) |
| **Orchard** (MSR+Columbia+UIUC, arXiv 2605.15040) | SWE: 107K distilled trajectories; GUI: 0.4K distilled + 2.2K tasks; Claw: 0.2K tasks | env-as-a-service; credit-assignment SFT on *productive segments of unresolved trajectories*; Balanced Adaptive Rollout; multi-turn GRPO (trajectory-level reward broadcast to assistant tokens, obs masked, asymmetric PPO, no 1/T norm); the 292K→15.6K 5-stage task filter |
| **τ-bench / TAU-bench** (Sierra; Yao, Shinn, Razavi, Narasimhan — ICLR 2025) | benchmark, not a corpus | the **user SIMULATOR**: "usable" = surviving a realistic simulated user + following domain policies; stateful DB-state verification |
| **OpenHands** (Xingyao Wang et al., ICLR 2025) | CodeActInstruct 7k; SWE-Gym; critic-rubrics 2026 (openhands-critic-4b) | real expert trajectories at scale; the open agent platform |
| **CAI corpus** (arXiv 2605.28146) | 230,935 expert-operator trajectories, 18.07 TB | "the corpus is a trace of every step the agent takes" — trajectory-level logs as the training substrate; weekly-retraining cadence; the agent's own trajectories feed back into the corpus |
| **Hermes 3** (teknium, Quesnelle, Chen Guang) | ~270M tokens SFT | 69% output tokens (INPUT masked); the data-filtering craft (Lambert: "much better at data filtering"); DPO with LoRA |

## The individual deep-dives

### 1. Karan Malhotra (@karan4d) — the trajectory-factory
Nous co-founder (2023, with Jeffrey Quesnelle). Lead of the Hermes 4
technical report. The recipe he published (the only open 5M-scale agentic
post-training corpus recipe):
- **DataForge**: pre-training seed (DCLM + FineWeb, recency-biased, ModernBERT
  semantic dedup at 0.7 cosine, LLM-judge filtering) → graph of struct→struct
  maps with PDDL pre/postconditions → random DAG walks → transformed passage
  → instruction (contextual or standalone, PersonaHub-like) → specialized
  answer generator → rubric judge (different weights than the answer model)
  → iterate or discard. **Every graph nests** (single source/single target) →
  higher-order graphs. **Train on the intermediate LLM calls too.**
- **Atropos** (the RL envs): format compliance decoupled from semantics (binary
  reward); RLVR-IFEval constraints; Internbootcamp → 70K rejection-sampled
  trajectories from ~1,000 reasoning tasks (multiple winning paths per task
  within a token budget, DeepHermes + larger teachers); Schema Adherence
  (dynamic Pydantic, programmatic validation reward); Tool-Use taxonomy DFS.
- The training side: loss-masking, length-control fine-tuning, efficient
  packing of heterogeneous data.

### 2. teknium / Dakota Mahan — the data-filtering craft
teknium (the Hermes trainer since the beginning; @Teknium1) + Dakota Mahan
(@dmayhem93, Hermes 4 author). The Hermes 3 stack: one large SFT mix then DPO
(LoRA adapters); 69% of tokens are OUTPUT tokens — the input is masked, only
the assistant's next-token prediction trains. Nathan Lambert's independent
read (interconnects.ai): Hermes's real edge is DATA FILTERING, not scale.
The reserved-token agentic grammar (`<|begin_of_turn|>`-style delimiters,
`<tool_call>` etc.) made Hermes 3 "an excellent choice for agentic tasks"
out of the box.

### 3. Shunyu Yao (姚顺雨) — the user simulator
Princeton PhD → OpenAI ("I study agents"). τ-bench (with Noah Shinn, Pedram
Razavi, Karthik Narasimhan; ICLR 2025): the missing third vertex of
tool-agent-USER. An LLM-based user simulator driven by per-task instructions
produces realistic user utterances; the agent must follow domain policy
documents and use the tools; the evaluation compares the DATABASE STATE after
the conversation to the annotated goal state — objective, stateful, faithful.
The principle: an agentic LLM is only "usable" if it survives a realistic
user, and the user simulator is a data-GENERATOR, not just an eval.

### 4. Xingyao Wang — real expert trajectories at scale
All Hands AI co-founder/CAIO; OpenDevin→OpenHands (ICLR 2025, with Boxuan Li,
Graham Neubig, Heng Ji, et al.). CodeAct (arXiv 2402.01030): executable-code
actions (Python/JS/bash as the action space) → CodeActInstruct (7k multi-turn);
SWE-Gym (2024, RL environments over real GitHub issues); 2026 critic-rubrics
(openhands-critic-4b: learning to verify AI-generated code). The message:
the best agent data is REAL task execution with verifiable outcomes
(unit tests, issue resolution).

### 5. Baolin Peng + Jianfeng Gao — the environment and the RL recipe
Orchard (MSR, with Wenlin Yao, Qianhui Wu, Hao Cheng, + Xiao Yu (Columbia),
Rui Yang (UIUC), Alessandro Sordoni, Xingdi Yuan, Yelong Shen, Pengcheng He,
Tong Zhang, Zhou Yu). Two mechanisms that matter for us:
- **Credit-assignment SFT**: learn from the PRODUCTIVE SEGMENTS of even
  UNRESOLVED trajectories (partial-credit supervision on failed traces).
- **Multi-turn trajectory-level GRPO**: sample G trajectories per task,
  compute group-relative advantage from the trajectory-level reward, broadcast
  it to every assistant token across all turns, mask observation/environment
  tokens out of the loss, asymmetric PPO clipping (0.2/0.28), NO per-trajectory
  1/T normalization (so longer harder tasks are not down-weighted).
- The numbers: 67.5% SWE-bench Verified (30B-A3B), 68.4% avg GUI (4B!), all
  from 0.4K-107K distilled trajectories. Quality + environment fidelity > raw scale.

## The 7-hop chain (all real, credited)

1. **Karan Malhotra** — Nous co-founder; the Hermes 4 5.1M-sample corpus recipe (DataForge + Atropos) — arXiv 2508.18255.
2. → **Jeffrey Quesnelle / Dakota Mahan / teknium** — the Hermes lineage (Quesnelle founded Nous 2023; teknium trained Hermes since 1; Hermes 3's masking + filtering craft; Hermes 4 authors).
3. → **Nathan Lambert** (Ai2) — the independent read: Hermes's edge is data FILTERING; the 69%-output-token masking; his interconnects analysis of the open post-training stacks.
4. → **Xingyao Wang** — OpenHands/CodeAct/SWE-Gym: real expert trajectories with verifiable outcomes (ICLR 2025).
5. → **Shunyu Yao** — τ-bench: the user simulator (ICLR 2025) — the tool-agent-USER third vertex; the stateful DB-state eval.
6. → **Baolin Peng / Jianfeng Gao** — Orchard (MSR): env-as-a-service, credit-assignment SFT, trajectory-level multi-turn GRPO (arXiv 2605.15040).
7. → **Tri Dao** — already in the ledger (FlashAttention, Gram-NS — the compute substrate the RL loop runs on); the verifiable-reward RL lineage that both Hermes (Atropos) and Orchard (GRPO) descend from.

## The convergence

**A usable agentic LLM is trained on a corpus that is MADE, not found, and
its quality is set by three things: expert demonstration, verifiable reward,
and a faithful user+environment simulator.** Orchard's 2.6k tasks beating
100×-larger corpora; Hermes's 5.1M made-samples vs. raw internet text; the
CAI corpus's "a trace of every step" doctrine. The scoreboard for a small
model (WuBu-35M):

1. **Made data beats scraped data** — DataForge-style synthesis (graph
   composition, judge≠answer weights, train on intermediate calls).
2. **Masking is a curriculum** — the assistant's tokens only (69% output);
   the observation tokens masked (Orchard); loss-masking + packing (Hermes 4).
3. **Verifiable reward beats preference ratings** — DB-state (τ-bench),
   format validity (Atropos), unit tests (OpenHands/SWE), trajectory-level
   GRPO with the group-relative advantage (Orchard).
4. **Partial credit on failures** — credit-assignment SFT on unresolved
   trajectories; rejection sampling keeps the winning paths.
5. **The user simulator is a data generator** — τ-bench-style simulators
   produce the "usable" distribution: realistic users, policy documents,
   stateful outcomes.
6. **The environment is a service** — sandbox lifecycle, command exec, file
   I/O, network policy as reusable primitives (Orchard Env); the corpus is
   the trace of every step (CAI).

## The closes into the WuBu-35M pipeline (this wave)

- `tools/wubu_traj_sft.c` — the trajectory→masked-observation-SFT converter
  (the Hermes 69%-output + Orchard obs-masking principle, own C11).
- `src/wubu_traj_grpo.c` — the multi-turn trajectory-level GRPO (the Orchard
  recipe: group-relative advantage, obs masking, asymmetric PPO clipping,
  no-1/T) with a finite-difference-verified advantage.
- `tools/wubu_user_sim.c` — the τ-bench-style user simulator harness for
  generating WuBu's own agentic data.
- The corpus-mix recipe (doc): expert trajectories + simulator rollouts +
  code traces + the 4.44B pretraining stream.

Archive: full sources in the wubuos compendium `05-sources/`
(hermes-4-technical-report.md, orchard-agentic-modeling.md); registered in
the MASTER-INDEX. The agentic-corpus avenue bank (1000 gaps) is generated
from this synthesis.
