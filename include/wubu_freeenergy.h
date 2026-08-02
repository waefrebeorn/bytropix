/*
 * wubu_freeenergy.h -- Predictive coding / free energy / active
 * inference (Theme IN). C11.
 *
 * Convergence (Friston free-energy principle; predictive-coding
 * hierarchies; expected free energy for policy selection):
 *   - IN01 Prediction-error layers: top-down predictions vs bottom-up
 *         errors -- e = x - mu_hat (the predictive-coding update).
 *   - IN02 Variational free energy: F = accuracy - complexity
 *         (the world-model's fit minus its cost).
 *   - IN03 Active inference: policy selection by the EXPECTED free
 *         energy -- p(pi) = softmax(-gamma * G(pi)).
 *   - IN04 Precision weighting: the precision-scaled prediction errors
 *         (pi * e; high precision = trusted error channels).
 *   - IN05 The perception-action loop: perception minimizes the
 *         prediction error; action minimizes the expected free energy.
 *   - IN06 Epistemic value: the information-gain policy bonus
 *         (the curiosity / exploration term).
 *   - IN07 The operator: free-energy-gated model selection (pick the
 *         world-model with the lowest free energy under a complexity
 *         budget).
 */
#ifndef WUBU_FREEENERGY_H
#define WUBU_FREEENERGY_H

/* IN01: the prediction error of one scalar channel. */
float wubu_fe_pred_error(float x, float mu_hat);

/* IN04: the precision-weighted error. */
float wubu_fe_precision_weight(float error, float precision);

/* IN02: the variational free energy of a model.
 * log_likelihood = the accuracy (more negative = better fit);
 * complexity = the KL from the prior (>= 0). F = -log_lik +
 * complexity. Returns F (>= 0 for well-formed inputs). */
float wubu_fe_free_energy(float log_likelihood, float complexity);

/* IN03: the expected free energy of a policy (negative = good).
 * pragmatic = -E[log p(o|s)] under the policy (the goal attainment);
 * epistemic = the expected information gain (the reduction in the
 * state uncertainty). G = pragmatic + epistemic. */
float wubu_fe_expected_free_energy(float pragmatic, float epistemic);
/* The policy prior: p(pi) = softmax(-gamma * G). out receives the
 * probabilities (n entries); returns 0. */
int wubu_fe_policy_prior(const float *G, int n, float gamma, float *out);

/* IN05: the perception-action update -- one perception step: the new
 * prediction = prediction + lr * precision * error. */
float wubu_fe_percept_step(float mu_hat, float error, float precision, float lr);

/* IN06: the epistemic value of an action = the expected reduction in
 * the state uncertainty (the info gain). */
float wubu_fe_epistemic_value(float uncertainty_before, float uncertainty_after);

/* IN07: the operator -- free-energy-gated model selection.
 * Returns the index of the model with the minimum free energy among
 * those with complexity <= max_complexity; -1 if none. */
int wubu_fe_pick_model(const float *fe, const float *complexity,
                       int n, float max_complexity);

#endif
