/*
 * wubu_dqn.c -- DQN / Q-learning (value-based) (GG05). C11.
 *
 * Convergence (Q-learning / DQN / experience replay 7-hop):
 *   - GG05: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]. Off-policy TD(0).
 *     Experience replay breaks temporal correlation. At home: config=state,
 *     sweep-choice=action, Q predicts tok_s; operator picks argmax Q → greedy
 *     optimal config without policy gradient. The DGM archive stores replay.
 */
#include "wubu_dqn.h"
#include <math.h>
#include <string.h>

static double rng_f(unsigned *s) {
    *s = (*s * 1103515245U + 12345U) & 0x7fffffff;
    return (double)(*s) / (double)0x7fffffff;
}

int wubu_dqn_init(wubu_dqn_t *dqn, int n_states, int n_actions, unsigned seed) {
    if (!dqn || n_states <= 0 || n_actions <= 0) return -1;
    memset(dqn, 0, sizeof(*dqn));
    dqn->n_states = n_states; dqn->n_actions = n_actions;
    dqn->gamma = 0.99; dqn->alpha = 0.1;
    unsigned s = seed ? seed : 11;
    for (int i = 0; i < n_states; i++)
        for (int a = 0; a < n_actions; a++)
            dqn->Q[i][a] = (rng_f(&s) - 0.5) * 0.01;
    return 0;
}

int wubu_dqn_update(wubu_dqn_t *dqn, int s, int a, double r, int s2) {
    if (!dqn || s < 0 || s >= dqn->n_states || a < 0 || a >= dqn->n_actions) return -1;
    if (s2 < 0 || s2 >= dqn->n_states) return -1;
    double max_q = -1e300;
    for (int a2 = 0; a2 < dqn->n_actions; a2++)
        if (dqn->Q[s2][a2] > max_q) max_q = dqn->Q[s2][a2];
    double target = r + dqn->gamma * max_q;
    dqn->Q[s][a] += dqn->alpha * (target - dqn->Q[s][a]);
    return 0;
}

int wubu_dqn_replay_store(wubu_dqn_t *dqn, int s, int a, double r, int s2) {
    if (!dqn || dqn->buf_n >= WUBU_DQN_BUF) return -1;
    dqn->buf_s[dqn->buf_head] = s;
    dqn->buf_a[dqn->buf_head] = a;
    dqn->buf_r[dqn->buf_head] = r;
    dqn->buf_s2[dqn->buf_head] = s2;
    dqn->buf_head = (dqn->buf_head + 1) % WUBU_DQN_BUF;
    if (dqn->buf_n < WUBU_DQN_BUF) dqn->buf_n++;
    return 0;
}

int wubu_dqn_replay_train(wubu_dqn_t *dqn, int batch_size) {
    if (!dqn || batch_size <= 0) return -1;
    int n = (batch_size < dqn->buf_n) ? batch_size : dqn->buf_n;
    for (int i = 0; i < n; i++) {
        int idx = (dqn->buf_head - 1 - i + WUBU_DQN_BUF) % dqn->buf_n;
        wubu_dqn_update(dqn, dqn->buf_s[idx], dqn->buf_a[idx], dqn->buf_r[idx], dqn->buf_s2[idx]);
    }
    return 0;
}

int wubu_dqn_greedy(const wubu_dqn_t *dqn, int s) {
    if (!dqn || s < 0 || s >= dqn->n_states) return -1;
    int best = 0; double best_q = -1e300;
    for (int a = 0; a < dqn->n_actions; a++)
        if (dqn->Q[s][a] > best_q) { best_q = dqn->Q[s][a]; best = a; }
    return best;
}
