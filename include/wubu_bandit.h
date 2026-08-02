/*
 * wubu_bandit.h -- Thompson Sampling / multi-armed bandits (FF06).
 */
#ifndef WUBU_BANDIT_H
#define WUBU_BANDIT_H

#define WUBU_BANDIT_MAX_ARMS 32

typedef struct {
    int n_arms;
    double prior_alpha[WUBU_BANDIT_MAX_ARMS];  /* Beta prior α */
    double prior_beta[WUBU_BANDIT_MAX_ARMS];   /* Beta prior β */
    int   pulls[WUBU_BANDIT_MAX_ARMS];
    int   rewards[WUBU_BANDIT_MAX_ARMS];
} wubu_bandit_t;

/* Initialize with uniform Beta(1,1) prior. */
int  wubu_bandit_init(wubu_bandit_t *b, int n_arms);
/* Thompson sample: draw from posterior Beta(α+s, β+f) for each arm, return argmax. */
int  wubu_bandit_sample(const wubu_bandit_t *b, unsigned *seed);
/* Update posterior after observing reward (1=success, 0=failure). */
int  wubu_bandit_update(wubu_bandit_t *b, int arm, int reward);

#endif