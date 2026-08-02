/*
 * wubu_hopfield2.h -- Hopfield frontier extensions (Theme IP). C11.
 * The mechanisms beyond wubu_hopfield's basic retrieval.
 *
 * Convergence (continuous-time Hopfield 2502.10122; dynamic-manifold
 * 2506.01303; federated many-to-one 2603.19902; spectral capacity):
 *   - Continuous-time memory dynamics (an RK4-integrated memory ODE)
 *   - Dynamic-manifold reorganization (context-modulated patterns)
 *   - Federated many-to-one (heteroassociative cue -> output binding)
 *   - Spectral-capacity accounting + pattern-separation metric
 *   - Memory-write scheduling + rehearsal consolidation
 *   - Beta annealing, cue-denoising precision, decay scheduling
 *   - Context-dependent recall gating, partial-cue retrieval
 *   - Interference monitor + re-orthogonalization repair
 *   - Episodic time-tagged memory + forgetting-curve integration
 *   - Replay scheduling + cue-to-tool selection memory
 */
#ifndef WUBU_HOPFIELD2_H
#define WUBU_HOPFIELD2_H

#include <stdint.h>

/* IP21/IP01: one RK4 step of the continuous-time memory dynamics.
 * state = the memory state (dim floats); field = the ODE field (dim);
 * dt = the step. Returns the new state into out (may alias state). */
int wubu_hf_rk4_step(const float *state, const float *field, int dim,
                     float dt, float *out);

/* IP02: dynamic-manifold reorganization. A context vector modulates the
 * stored pattern: p' = normalize(p + ctx_gain * context). */
int wubu_hf_manifold_shift(const float *pattern, const float *context,
                           int dim, float gain, float *out);

/* IP03/IP14: federated many-to-one binding (heteroassociative).
 * Stores cue -> output pairs; retrieval = out for the closest cue.
 * Returns 1 if a bound output was found, 0 otherwise. */
int wubu_hf_federated_bind(float *cues, float *outs, int n_pairs, int dim,
                           const float *cue, int *best, float *out);

/* IP04/IP18: spectral-capacity accounting. Capacity estimate from the
 * spectral norm (max singular-ish scale) of the pattern matrix:
 * C = exp(alpha * dim) * (1 - spectral_sat). */
float wubu_hf_spectral_capacity(int dim, float alpha, float spectral_sat);

/* IP08: pattern-separation metric (normalized min pairwise distance
 * over the stored patterns). */
float wubu_hf_separation(const float *X, int n_pat, int dim);

/* IP06: memory-write scheduling -- should a pattern be stored?
 * Store when the novelty clears the threshold (see wubu_ev_novelty). */
int wubu_hf_should_store(float novelty, float novelty_thresh, int capacity_left);

/* IP09/IP26: rehearsal consolidation -- a rehearsed pattern's weight
 * grows; returns the updated weight. */
float wubu_hf_rehearse(float weight, float reward, float alpha);

/* IP07: beta annealing schedule (sharp-to-flat): beta(t) =
 * beta_max - (beta_max - beta_min) * (t / t_max). */
float wubu_hf_beta_anneal(float beta_max, float beta_min, int t, int t_max);

/* IP11: cue-denoising precision: the recall quality from a noisy cue
 * as a function of the cue SNR (0..1 = clean). */
float wubu_hf_denoise_quality(float snr, float beta);

/* IP12: decay scheduler: halflife adapted by utility (higher utility
 * -> longer halflife). */
float wubu_hf_decay_schedule(float base_halflife, float utility);

/* IP13: context-dependent recall gating: the gate = similarity of the
 * recall context to the stored pattern's context. */
float wubu_hf_context_gate(const float *ctx, const float *pat_ctx, int dim);

/* IP16: partial-cue retrieval: the overlap of a partial cue with a
 * stored pattern (fraction of the dims known). */
float wubu_hf_partial_overlap(const float *cue, const float *pattern,
                              int dim, const uint8_t *known);

/* IP10: interference monitor: crosstalk between two stored patterns. */
float wubu_hf_interference(const float *a, const float *b, int dim);

/* IP20: interference repair -- re-orthogonalize b against a
 * (subtract the projection). */
int wubu_hf_orthogonalize(const float *a, int dim, float *b);

/* IP19/IP25: episodic time-tagged memory: the effective weight of a
 * pattern at age t (forgetting curve: weight * 2^(-age/halflife) and
 * the time-tag is kept with the pattern). */
float wubu_hf_episodic_weight(float base, int age, float halflife);

/* IP29: cue-to-tool selection memory: pick the tool whose stored cue
 * best matches the request cue; returns the tool index. */
int wubu_hf_tool_select(const float *tool_cues, int n_tools, int dim,
                        const float *request);

#endif
