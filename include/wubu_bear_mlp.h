/*
 * wubu_bear_mlp.h  --  BearRL MLP Policy Forward (CPU, C11 fallback).
 *
 * Self-contained C11 port of the CUDA policy_forward_kernel from
 * WuBuOS/src/bear/bear_kernels.cu, adapted for CPU + Windows.
 *
 * Provides a pure-C11 MLP forward pass for BearRL policy networks,
 * plus GAE advantage computation. No CUDA, no Tensor Cores.
 * Can be used as a CPU fallback when CUDA is unavailable.
 *
 * SPDX-License-Identifier: WaefreBeorn-UMV3
 */
#ifndef WUBU_BEAR_MLP_H
#define WUBU_BEAR_MLP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Policy MLP forward pass.
 *
 * obs:    [batch, obs_dim]     input observations
 * h_in:   [batch, hidden_dim]  (optional, can be NULL for non-recurrent)
 * W1:     [hidden_dim, obs_dim]  layer 1 weights (row-major)
 * b1:     [hidden_dim]           layer 1 bias
 * W2:     [out_dim, hidden_dim]  layer 2 weights (row-major)
 *   where out_dim = act_dim + 1 + 1 + hidden_dim
 *   (action means + shared logstd + value + next hidden)
 * b2:     [out_dim]             layer 2 bias
 *
 * Outputs:
 * actions:    [batch, act_dim]     action means
 * logprobs:   [batch]              log probabilities (simplified)
 * values:     [batch]              value estimates
 * h_out:      [batch, hidden_dim]  next hidden state (ReLU of layer 1)
 *
 * Returns 0 on success, -1 on invalid args.
 */
int wubu_bear_mlp_forward(const float *obs,
                          const float *h_in,
                          const float *W1, const float *b1,
                          const float *W2, const float *b2,
                          float *actions, float *logprobs,
                          float *values, float *h_out,
                          int batch, int obs_dim, int hidden_dim, int act_dim);

/* GAE (Generalized Advantage Estimation) computation.
 * CPU port of bear_gae_kernel.
 *
 * rewards:   [T*B]  rewards
 * dones:     [T*B]  done flags (0 or 1)
 * values:    [T*B]  value estimates
 * advantages: [T*B] output advantages
 * returns:   [T*B] output returns
 *
 * gamma: discount factor
 * gae_lambda: GAE lambda
 * T: number of time steps
 * B: number of environments (batch)
 */
int wubu_bear_gae(const float *rewards, const uint8_t *dones,
                  const float *values, float *advantages, float *returns,
                  int T, int B, float gamma, float gae_lambda);

/* Simplified cartpole physics step (single pole).
 * CPU port of bear_npole_step_kernel.
 */
int wubu_bear_cartpole_step(float *x, float *x_dot,
                            float *theta, float *theta_dot,
                            float *reward, uint8_t *done,
                            const float *force,
                            int num_envs,
                            float pole_mass, float pole_length,
                            float gravity, float cart_mass, float dt,
                            float force_mag, float angle_threshold,
                            float pos_threshold);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_BEAR_MLP_H */
