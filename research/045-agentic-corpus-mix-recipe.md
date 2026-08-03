# The WuBu-35M agentic corpus-mix recipe (2026-08-04)

The corpus side of the standing loop, on top of the 4.44B pretraining
stream (FineMath 3.778B + OpenMath 0.664B + cosmopedia 0.27B, staged at
`/home/wubu/models/corpus/`). The agentic layer is MADE data (the 044
convergence): expert demonstration + verifiable reward + a faithful user
simulator. A 35M model cannot absorb 30k-token traces -- the budget is
small, the ratios matter.

## The three agentic streams

| Stream | Source doctrine | For WuBu-35M | Size target |
|---|---|---|---|
| A. Synthesized tool-use + format | Hermes-4 DataForge (graph synthesis, judge != answer, train on intermediates) | single-tool tasks with the <tool_call> grammar; format-valid (JSON); <think>-lite traces | ~40-60M tokens |
| B. Distilled + credited trajectories | Orchard (teacher distillation, credit-assignment SFT on productive segments) | short verified trajectories from the user-sim + env rollouts; the credit mask via `wubu_credit_mask` | ~30-50M tokens |
| C. Real expert traces | OpenHands/CodeAct | code-execution traces with unit-test outcomes (the smallest stream -- the most expensive to make) | ~10-20M tokens |

Total agentic SFT: ~80-130M tokens on top of the 4.44B pretraining (the
~2-3% agentic layer -- the Hermes-3 ratio was ~2% of the base at 270M/13B).

## The RL stage (after the SFT)

- The trajectory-level GRPO (`wubu_traj_grpo`, FD-verified): group of G=8
  rollouts per task, trajectory-level reward (the DB-state verifier
  `wubu_db_verify` / format validity), the advantage broadcast to the
  assistant tokens, the obs masked, the asymmetric PPO clip 0.2/0.28, no
  1/T normalization.
- The reward: `wubu_db_reward` (stateful, objective) + the format binary
  (the Atropos Answer-Format doctrine).
- The user simulator (`wubu_user_sim`) generates the rollout tasks: goals +
  personas + policy; the agent's actions update the state; the verifier
  scores the end state.

## The pipeline order (the standing loop)

1. Pretrain on the 4.44B stream (the current seed checkpoints).
2. SFT on the agentic mix (A + B + C), the masked-observation format
   (`wubu_traj_sft`: assistant/think/tool_call train; obs/context masked).
3. RL with the trajectory-level GRPO + the DB-state reward.
4. Diagnose (the amoeba: plateau detection, per-group grad norms) -- the
   failures become the next tasks (the CAI feedback-cadence doctrine).
5. Archive + repeat.

## The measured numbers that anchor this

- Orchard-GUI: 68.4% avg (WebVoyager/Online-Mind2Web/DeepShop) from 0.4K
  distilled + 2.2K tasks on a 4B backbone -- quality + fidelity >> scale.
- Hermes 4: 5.1M made samples / 19B tokens; 69% output tokens (the mask).
- The CAI corpus: 230,935 trajectories, weekly-retraining cadence -- the
  corpus is a trace of every step; the deployed agent's traces feed back.
