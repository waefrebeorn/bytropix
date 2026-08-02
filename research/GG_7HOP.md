# Reinforcement Learning: Policy Gradients, Actor-Critic, PPO, Value-Based — 7-hop KB sweep
## GG axis: reward-driven policy learning for the AGI-OS operator (at home, C11)

> Each stone seeds the next hop. Target: give WuBuOS a real RL substrate so
> recursive_optimize's blind sweeps become reward-driven policy learning.

## Hop 1: REINFORCE (policy gradient)
Policy gradient theorem: ∇J(θ) = E[Σ_t ∇log π(a_t|s_t) · G_t] where G_t = Σ γ^t R_t
is the discounted return. REINFORCE is on-policy, Monte-Carlo (uses full
episode return). High variance → needs baseline.
At home: the AGI-OS operator's config choices are actions; tok_s is reward.
REINFORCE learns a policy π(config) that maximizes expected tok_s.

## Hop 2: Variance reduction (baseline / advantage)
REINFORCE with baseline: subtract b(s_t) from return: ∇J = E[∇log π · (G_t - b)].
Baseline (e.g. value function V(s)) reduces gradient variance without bias.
At home: a learned baseline V(config) centers the returns → stable learning.

## Hop 3: Actor-Critic (TD baseline)
Actor-Critic = policy gradient (actor) + learned value baseline (critic) via TD.
Advantage A(s,a) = Q(s,a) - V(s) = r + γV(s') - V(s) (TD error δ).
At home: critic learns V(config); actor updates using TD advantage → lower
variance than Monte-Carlo REINFORCE, online (no full episode needed).

## Hop 4: PPO (clipped surrogate objective)
PPO: ratio r(θ) = π_θ(a|s) / π_θ_old(a|s). Clipped objective:
  L = min(r·A, clip(r, 1-ε, 1+ε)·A).  ε≈0.2.
Prevents destructive large policy updates (trust region, first-order).
At home: the operator's policy updates are clipped → stable config optimization
without blowing up the sweep (which costs real wall-clock time).

## Hop 5: Value-based (Q-learning / DQN)
Q-learning: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]. TD(0) off-policy.
DQN: experience replay (break correlation) + target network (stable targets).
At home: config = state, sweep-choice = action, Q(config,choice) predicts tok_s.
The operator picks argmax Q → greedy optimal config without policy gradient.

## Hop 6: Integration — hybrid RL operator
The RL stack for the AGI-OS loop:
  1. REINFORCE initializes policy π(config)                     [GG01 wubu_reinforce]
  2. Baseline V(s) reduces variance                            [GG02 wubu_policy]
  3. Actor-Critic: TD advantage from critic                    [GG03 wubu_actor_critic]
  4. PPO: clipped stable updates                               [GG04 wubu_ppo]
  5. DQN/Q-learning: value-based alternative                   [GG05 wubu_dqn]
  6. Value iteration / Bellman backup                          [GG06 wubu_value]
  7. Unified policy struct (π, V, Q share interface)           [GG07 integration]

## Hop 7: Closed loop with recursive_optimize
Replace blind hill-climbing with: observe state (config + tok_s) → RL agent
picks next config → reward = tok_s gain → update policy/value → repeat.
The DGM archive (AX01) stores trajectories; continual learning (BB) prevents
forgetting the policy when the world shifts. This is the reward-driven core
the whole AGI-OS substrate was missing.

## Gap mapping
- GG01 REINFORCE (policy gradient, Monte-Carlo) `wired` (wubu_reinforce.c)
- GG02 Baseline / variance reduction `wired` (wubu_policy.c)
- GG03 Actor-Critic (TD advantage) `wired` (wubu_actor_critic.c)
- GG04 PPO (clipped surrogate) `wired` (wubu_ppo.c)
- GG05 DQN / Q-learning (value-based) `wired` (wubu_dqn.c)
- GG06 Value iteration / Bellman backup `wired` (wubu_value.c)
- GG07 Unified policy/value interface + integration `wired` (test_gg.c)
