/**
 * qlearner.c — Q-Learning LR Controller (meta-optimizer)
 *
 * Tunes the learning rate based on training dynamics.
 * Reward = 1/(loss + eps): lower loss = higher reward.
 * States discretize the loss magnitude into 10 bins.
 * Actions: decrease/hold/increase LR by *0.8/*1.0/*1.2.
 *
 * Based on QLearnerLR from Wubu_Clockwork_Perceptron.py
 */
#include "qlearner.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

void qlearner_init(qlearner_t *ql) {
    memset(ql->table, 0, sizeof(ql->table));
    ql->state = 0;
    ql->action = 1;     // hold
    ql->lr = QL_LR_INIT;
    ql->step_count = 0;
}

float qlearner_step(qlearner_t *ql, float loss) {
    // Reward: stability = low loss
    float reward = 1.0f / (loss + 1e-5f);

    // State: loss magnitude bin (0-9)
    int next_state = (int)(loss * 20.0f);
    if (next_state < 0) next_state = 0;
    if (next_state >= QL_N_STATES) next_state = QL_N_STATES - 1;

    // Find max Q for next state
    float max_next = ql->table[next_state][0];
    for (int a = 1; a < QL_N_ACTIONS; a++) {
        if (ql->table[next_state][a] > max_next)
            max_next = ql->table[next_state][a];
    }

    // Q-learning update: Q(s,a) += alpha * (reward + gamma * max(Q(s')) - Q(s,a))
    float target = reward + QL_GAMMA * max_next;
    float td_error = target - ql->table[ql->state][ql->action];
    ql->table[ql->state][ql->action] += QL_ALPHA * td_error;

    // Epsilon-greedy: choose action for next state
    int action;
    if (((float)rand() / RAND_MAX) < QL_EPSILON) {
        action = rand() % QL_N_ACTIONS;  // explore
    } else {
        action = 0;
        float best_q = ql->table[next_state][0];
        for (int a = 1; a < QL_N_ACTIONS; a++) {
            if (ql->table[next_state][a] > best_q) {
                best_q = ql->table[next_state][a];
                action = a;
            }
        }
    }

    // Apply action to LR
    if (action == 0)      ql->lr *= 0.8f;   // decrease
    else if (action == 2) ql->lr *= 1.2f;   // increase
    // action == 1: hold

    // Clamp LR
    if (ql->lr < QL_LR_MIN) ql->lr = QL_LR_MIN;
    if (ql->lr > QL_LR_MAX) ql->lr = QL_LR_MAX;

    ql->state = next_state;
    ql->action = action;
    ql->step_count++;

    return ql->lr;
}
