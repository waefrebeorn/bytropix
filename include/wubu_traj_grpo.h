/* wubu_traj_grpo.h -- the multi-turn trajectory-level GRPO (the Orchard
 * recipe core, own C11): group-relative advantage over G trajectories,
 * the masked advantage-weighted NLL loss (observation tokens masked out),
 * and the asymmetric-PPO-clipped ratio variant. FD-verifiable: the
 * gradients are checked against finite differences in the test. */
#ifndef WUBU_TRAJ_GRPO_H
#define WUBU_TRAJ_GRPO_H

/* Compute the trajectory-level GRPO loss + gradient.
 * logp [G*T]: the per-token log-probs of the policy (row g = trajectory g).
 * mask [G*T]: 1 = the token trains (assistant tokens), 0 = masked
 *             (observation/context/user tokens -- the Orchard doctrine).
 * r [G]: the trajectory-level rewards.
 * clip_lo / clip_hi: the asymmetric PPO clip range (e.g. 0.2 / 0.28);
 *                    0 disables the clipping (plain advantage NLL).
 * old_logp [G*T] or NULL: the reference log-probs for the ratio clipping.
 * eps: the std-floor (avoid div-by-zero on equal rewards).
 * out loss (scalar), grad [G*T] dL/dlogp (may be NULL).
 * Returns 1 on success. */
int wubu_traj_grpo(const float *logp, const float *mask, const float *r,
                   int G, int T, float clip_lo, float clip_hi,
                   const float *old_logp, float eps,
                   float *loss, float *grad);

#endif
