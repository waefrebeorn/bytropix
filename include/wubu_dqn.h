/*
 * wubu_dqn.h -- DQN / Q-learning (value-based) (GG05).
 */
#ifndef WUBU_DQN_H
#define WUBU_DQN_H

#define WUBU_DQN_MAX_STATE 64
#define WUBU_DQN_MAX_ACTIONS 16
#define WUBU_DQN_BUF 1024

typedef struct {
    int n_states;
    int n_actions;
    double Q[WUBU_DQN_MAX_STATE][WUBU_DQN_MAX_ACTIONS];
    double gamma;
    double alpha;       /* learning rate */
    /* Experience replay buffer */
    int   buf_s[WUBU_DQN_BUF];
    int   buf_a[WUBU_DQN_BUF];
    double buf_r[WUBU_DQN_BUF];
    int   buf_s2[WUBU_DQN_BUF];
    int   buf_n;
    int   buf_head;
} wubu_dqn_t;

/* Init Q-table (small random). */
int  wubu_dqn_init(wubu_dqn_t *dqn, int n_states, int n_actions, unsigned seed);
/* Q-learning update: Q(s,a) += α[r + γ max_a' Q(s',a') - Q(s,a)]. */
int  wubu_dqn_update(wubu_dqn_t *dqn, int s, int a, double r, int s2);
/* Experience replay: store + sample-batch update. */
int  wubu_dqn_replay_store(wubu_dqn_t *dqn, int s, int a, double r, int s2);
int  wubu_dqn_replay_train(wubu_dqn_t *dqn, int batch_size);
/* Greedy action (argmax Q). */
int  wubu_dqn_greedy(const wubu_dqn_t *dqn, int s);

#endif