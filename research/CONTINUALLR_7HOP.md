# Continual Learning + Catastrophic Forgetting Prevention — 7-hop KB sweep
## BB axis: the agent's ability to learn continuously without forgetting (at home)

> Each stone seeds the next hop. Target: map the continual-learning substrate
> that AGI-at-home is STILL missing.

## Hop 1: Continual learning / catastrophic forgetting
Catastrophic forgetting = NN abruptly loses prior task knowledge when learning new tasks.
Humans reuse/refine experience continuously. Three strategy categories:
  - Regularization (EWC, SI): protect important parameters with quadratic penalties.
    EWC uses Fisher Information Matrix (diagonal) for weight importance; recent work
    shows FIM causes gradient vanishing — Logits Reversal (LR) fixes this.
  - Architecture (Progressive Networks): dynamically expand network per task.
  - Replay (ER, A-GEM, DER++): store past samples in finite buffer, interleave.
    For LMs: experience replay (mixing old task samples) is most effective.
At home: our recursive_optimize sweeps params but has NO forgetting prevention — every
new sweep could overwrite the KV strategy that worked last time.

## Hop 2: Experience replay + reservoir sampling
Experience Replay (ER): store SARS tuples in finite-capacity buffer, sample mini-batches
to reduce sample correlation + improve stability. Memory architectures: FIFO, dual-buffer,
episodic memory, reservoir sampling (probabilistic replacement, statistically representative
for streaming). Non-uniform replay can outperform uniform. Key insight: reservoir sampling
gives O(1) per-item storage for infinite streams, perfect for at-home memory-bounded.
At home: our DGM archive IS a form of replay (196 nodes logged), but it's outcome-only
(KV config + tok/s), not trajectory replay. We need an actual replay buffer of
parameter states + outputs.

## Hop 3: Elastic Weight Consolidation (EWC) + Fisher Information
EWC (Kirkpatrick 2017): parameter importance = diagonal Fisher Information Matrix.
Loss = task_loss + λ·Σ F_i·(θ_i - θ*_i)². High-F = "stable synapses" (protected),
low-F = "plastic synapses" (free to adapt). Recent: EWC done right (logits reversal,
arXiv 2603.18596) — reversing logits during FIM calculation prevents gradient vanishing
+ redundant protection. Scaling to multiple tasks: accumulate Fisher across tasks.
At home: our recursive_optimize has no parameter-importance tracking — it sweeps 15 dims
blindly. EWC would protect the best-scoring dims from being overwritten.

## Hop 4: Synaptic Intelligence (SI) + path-integral importance
SI: importance = integral of gradient·parameter-change along the optimization path
(not just endpoint curvature like EWC). Captures parameters that played central roles
in the *path* to convergence, even if flat at the end. Hookean restoring force:
high-importance param accumulates λ×displacement opposing change. Different notion
from EWC (path vs. endpoint). At home: we could track the *trajectory* of each sweep
dim and protect dims that consistently contributed to performance gains.

## Hop 5: Task boundary detection (online, no task ID)
Unknown task boundary: the agent doesn't know when a task starts/ends. Methods:
  - OOD detection via softmax probabilities (detect when input distribution shifts).
  - Online boundary-free (no task boundaries assumed at all).
  - Task-free diversity-aware (no predefined tasks).
  - RL-based task identification (policy rewards model performance).
At home: when the AGI-OS operator detects a performance drop in `gen_text` (world-model
divergence), that's a task boundary — the environment distribution shifted. The agent
should detect this and trigger EWC consolidation BEFORE sweeping new dimensions.

## Hop 6: Knowledge distillation + replay hybrids (DER++)
DER++ (Dark Experience Replay++): combines experience replay with knowledge distillation.
Replay stored samples + distill the old model's outputs on new data. Prevents the
"stability-plasticity dilemma": too much stability → no new learning; too much plasticity
→ forgetting. Dark knowledge (soft labels) preserves the old model's output distribution.
At home: our operator re-runs the sweep but doesn't preserve old model outputs —
we just pick the best config. Distillation would let us keep a "teacher" snapshot
of the old best and use it as a soft target when sweeping new configs.

## Hop 7: Integration with AGI-OS substrate
The continual learning loop:
  1. World-model detects divergence (performance drop in gen_text)  [AG-04 world-model]
  2. → triggers task-boundary detection                             [BB03]
  3. → EWC consolidation: compute Fisher importance for best dims    [BB02]
  4. → lock important dims (capzero/safekern: read-only on stable) [capzero + BB02]
  5. → replay buffer of past configs + outputs                      [BB01]
  6. → new sweep runs on plastic dims only                         [recursive_optimize]
  7. → SI path tracking: update importance as sweeps run           [BB03]
  8. → distillation: teacher snapshot + soft targets               [BB04]
  9. → loopguard: prevent runaway forgetting                       [AG-01 loopguard]

Closed-loop: the agent learns continuously, remembers what matters, and the substrate
monitors its own forgetting.

## Gap mapping to WuBuOS substrate
- BB01 Experience replay buffer (ring + reservoir sampling) `wired` (wubu_replay.c)
- BB02 EWC consolidation (Fisher importance + quadratic penalty) `wired` (wubu_ewc.c)
- BB03 Task boundary detection (OOD via divergence signal) `wired` (wubu_taskbd.c)
- BB04 Knowledge distillation (teacher snapshot + KL soft targets) `wired` (wubu_distill.c)
- BB05 Plasticity/stability gating → integration with DGM archive + recursive_optimize

> Note: Full EWC on model weights requires weight-level access (model .gguf). At home,
> we implement parameter-IMPORTANCE tracking on the 15-dim sweep space instead — the
> same mechanism (protect important dims, allow plasticity in unimportant ones) applies
> at the config level. This is the "continual learning for the operator" analog.
