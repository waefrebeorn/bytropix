/*
 * wubu_policy.h -- Policy representation + baseline (GG02).
 */
#ifndef WUBU_POLICY_H
#define WUBU_POLICY_H

#define WUBU_POLICY_MAX_ACTIONS 16
#define WUBU_POLICY_MAX_STATE 64

typedef struct {
    int n_actions;
    /* Linear policy: logits = W · state + b. Softmax → action probs. */
    double W[WUBU_POLICY_MAX_ACTIONS][WUBU_POLICY_MAX_STATE];
    double b[WUBU_POLICY_MAX_ACTIONS];
    int   state_dim;
    /* Learned baseline (value) per state feature (mean return). */
    double baseline[WUBU_POLICY_MAX_STATE];
    int   baseline_n;
} wubu_policy_t;

/* Init policy with small random weights (seeded). */
int  wubu_policy_init(wubu_policy_t *p, int n_actions, int state_dim, unsigned seed);
/* Action probabilities given state (softmax over logits). */
int  wubu_policy_probs(const wubu_policy_t *p, const double *state, double *probs);
/* Sample an action from the policy (given probs + rng). */
int  wubu_policy_sample(const double *probs, int n_actions, unsigned *seed);
/* Update baseline estimate (exponential moving average of returns). */
int  wubu_policy_update_baseline(wubu_policy_t *p, double return_val, double lr);

#endif