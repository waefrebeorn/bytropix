#ifndef WUBU_QLEARNER_H
#define WUBU_QLEARNER_H

// Q-Learner LR Controller for meta-tuning learning rate
// States: 10 bins based on loss magnitude (loss * 20, clamped 0-9)
// Actions: 3 (0=decrease, 1=hold, 2=increase)
// Q-table [N_STATES][N_ACTIONS]
// Reward: 1/(loss + eps) — stability = low loss

#define QL_N_STATES  10
#define QL_N_ACTIONS 3
#define QL_EPSILON   0.1f
#define QL_GAMMA     0.9f
#define QL_ALPHA     0.1f   // Q-learning rate
#define QL_LR_INIT   0.05f
#define QL_LR_MIN    1e-5f
#define QL_LR_MAX    0.2f

typedef struct {
    float table[QL_N_STATES][QL_N_ACTIONS];  // Q-table
    int state;     // current state index
    int action;    // last action taken
    float lr;      // current learning rate (output)
    int step_count;
} qlearner_t;

// Initialize Q-learner (zeros table, state=0, action=1, lr=QL_LR_INIT)
void qlearner_init(qlearner_t *ql);

// Step: feed loss, get adjusted learning rate
// Returns the new learning rate after Q-learning update
float qlearner_step(qlearner_t *ql, float loss);

#endif // WUBU_QLEARNER_H
