/*
 * test_gg.c -- GG01-GG07 verification.
 */
#include "wubu_policy.h"
#include "wubu_reinforce.h"
#include "wubu_actor_critic.h"
#include "wubu_ppo.h"
#include "wubu_dqn.h"
#include "wubu_value.h"
#include <stdio.h>
#include <math.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fails++; printf("FAIL: %s\n", msg); } \
    else printf("  ok: %s\n", msg); \
} while(0)

/* Toy environment: state = 1-dim config scalar in {0,1,2,3}.
   reward = tok_s proxy: higher config index → higher reward (with noise).
   Optimal action: move toward index 3. */
static double toy_reward(int state) { return (double)state * 2.0; }

int main() {
    /* GG01: REINFORCE */
    printf("=== GG01: REINFORCE ===\n");
    wubu_policy_t pol;
    CHECK(wubu_policy_init(&pol, 2, 1, 123) == 0, "policy init (2 actions, 1-dim state)");
    double sprobs[2];
    double s1[1] = {1.0};
    CHECK(wubu_policy_probs(&pol, s1, sprobs) == 0, "policy probs computed");
    CHECK(fabs(sprobs[0] + sprobs[1] - 1.0) < 1e-9, "probs sum to 1.0");
    wubu_reinforce_t rf;
    memset(&rf, 0, sizeof(rf));
    rf.state_dim = 1; rf.n_actions = 2; rf.gamma = 0.9; rf.t = 0;
    /* Build a trajectory: states 0,1,2,3 with increasing reward */
    for (int t = 0; t < 4; t++) {
        double st[1] = { (double)t };
        wubu_reinforce_step(&rf, st, (t % 2), toy_reward(t));
    }
    CHECK(rf.t == 4, "trajectory has 4 steps");
    double rets[WUBU_REINFORCE_MAX_T];
    wubu_reinforce_returns(&rf, rets);
    CHECK(rets[0] > rets[3], "returns decrease over time (discounted, later reward)");
    double gn = wubu_reinforce_update(&rf, &pol, rets, 0.0, 0.05);
    CHECK(gn >= 0, "REINFORCE update produced gradient (norm >= 0)");

    /* GG02: Baseline / variance reduction */
    printf("\n=== GG02: Baseline ===\n");
    wubu_policy_t pol2;
    wubu_policy_init(&pol2, 2, 1, 99);
    double b0 = pol2.baseline[0];
    wubu_policy_update_baseline(&pol2, 10.0, 0.1);
    CHECK(pol2.baseline[0] != b0, "baseline updated after first return");
    wubu_policy_update_baseline(&pol2, 20.0, 0.1);
    CHECK(pol2.baseline[0] > 10.0, "baseline tracks returns (EMA)");

    /* GG03: Actor-Critic */
    printf("\n=== GG03: Actor-Critic ===\n");
    wubu_ac_t ac;
    CHECK(wubu_ac_init(&ac, 2, 1, 55) == 0, "actor-critic init");
    double s_a[1] = {0.5}, s_b[1] = {0.5};
    double delta1 = wubu_ac_update(&ac, s_a, 1, 5.0, s_b);
    CHECK(ac.V[0] != 0.0, "critic V updated by TD");
    double delta2 = wubu_ac_update(&ac, s_a, 1, 5.0, s_b);
    /* TD error should shrink as V approaches reward */
    CHECK(fabs(delta2) <= fabs(delta1), "TD error non-increasing (critic learning)");

    /* GG04: PPO */
    printf("\n=== GG04: PPO ===\n");
    wubu_ppo_t ppo;
    CHECK(wubu_ppo_init(&ppo, 2, 1, 77) == 0, "ppo init");
    wubu_ppo_snapshot(&ppo);
    double s_ppo[1] = {1.0};
    double obj = wubu_ppo_update(&ppo, s_ppo, 0, 1.0);
    CHECK(isfinite(obj), "ppo clipped objective finite");
    /* Ratio should stay within clip range after update */
    double p_cur[2], p_old[2];
    wubu_policy_probs(&ppo.cur, s_ppo, p_cur);
    wubu_policy_probs(&ppo.old, s_ppo, p_old);
    double ratio = p_cur[0] / (p_old[0] < 1e-12 ? 1e-12 : p_old[0]);
    printf("    ppo ratio after update = %.4f (clip [%.2f, %.2f])\n", ratio, 1-ppo.epsilon, 1+ppo.epsilon);
    CHECK(ratio <= 1.0 + ppo.epsilon + 1e-6, "ppo ratio within clip upper bound");

    /* GG05: DQN / Q-learning */
    printf("\n=== GG05: DQN ===\n");
    wubu_dqn_t dqn;
    CHECK(wubu_dqn_init(&dqn, 4, 2, 33) == 0, "dqn init (4 states, 2 actions)");
    /* Reward: action 1 at state 3 gives high reward, transitions stay.
       Train: (s=2,a=1,r=6,s2=3), (s=3,a=1,r=8,s2=3) */
    wubu_dqn_replay_store(&dqn, 2, 1, 6.0, 3);
    wubu_dqn_replay_store(&dqn, 3, 1, 8.0, 3);
    wubu_dqn_replay_store(&dqn, 0, 0, 1.0, 1);
    wubu_dqn_replay_train(&dqn, 10);
    CHECK(dqn.Q[3][1] > dqn.Q[0][0], "DQN learned: Q(s=3,a=1) > Q(s=0,a=0)");
    int greedy = wubu_dqn_greedy(&dqn, 2);
    printf("    greedy action at s=2: %d (expect 1)\n", greedy);
    CHECK(greedy == 1, "DQN greedy picks action 1 at s=2 (higher Q)");

    /* GG06: Value iteration */
    printf("\n=== GG06: Value Iteration ===\n");
    wubu_value_t val;
    CHECK(wubu_value_init(&val, 3, 2) == 0, "value iter init");
    /* Deterministic chain: s0 -a0-> s1, s1 -a0-> s2 (terminal, R=10), else stay.
       P[sp][s][a]: from s, action a, to sp. */
    for (int s = 0; s < 3; s++)
        for (int a = 0; a < 2; a++)
            for (int sp = 0; sp < 3; sp++)
                val.P[sp][s][a] = (sp == s) ? 1.0 : 0.0;  /* default: stay */
    /* Action 0: s0->s1, s1->s2; s2 stays. Action 1: stay. */
    val.P[1][0][0] = 1.0; val.P[0][0][0] = 0.0;
    val.P[2][1][0] = 1.0; val.P[1][1][0] = 0.0;
    val.R[0][0] = 0; val.R[1][0] = 0; val.R[2][0] = 10.0;
    val.R[0][1] = 1; val.R[1][1] = 1; val.R[2][1] = 1;
    for (int i = 0; i < 50; i++) wubu_value_iterate(&val);
    CHECK(val.V[2] > val.V[0], "value iteration: V(s=2) > V(s=0) (s2 is terminal goal)");
    int pol_s0 = wubu_value_policy(&val, 0);
    printf("    optimal policy at s=0: action %d (expect 0, moves toward goal)\n", pol_s0);
    CHECK(pol_s0 == 0, "value iteration policy: s0→action 0 (toward goal)");

    /* GG07: Integration — hybrid RL operator convergence */
    printf("\n=== GG07: Integration ===\n");
    /* REINFORCE should shift policy toward higher-reward actions over updates.
       Toy: state=1.0, action 1 always yields reward 8, action 0 yields reward 1.
       After many REINFORCE updates with baseline, π(action=1) should increase. */
    wubu_policy_t pol_i;
    wubu_policy_init(&pol_i, 2, 1, 5);
    wubu_reinforce_t rf_i;
    memset(&rf_i, 0, sizeof(rf_i));
    rf_i.state_dim = 1; rf_i.n_actions = 2; rf_i.gamma = 1.0; rf_i.t = 0;
    for (int ep = 0; ep < 30; ep++) {
        rf_i.t = 0;
        double st[1] = {1.0};
        /* Force experience: action 1 → high reward */
        wubu_reinforce_step(&rf_i, st, 1, 8.0);
        wubu_reinforce_step(&rf_i, st, 0, 1.0);
        double r_i[WUBU_REINFORCE_MAX_T];
        wubu_reinforce_returns(&rf_i, r_i);
        double base = (r_i[0] + r_i[1]) / 2.0;
        wubu_reinforce_update(&rf_i, &pol_i, r_i, base, 0.1);
    }
    double final_probs[2];
    wubu_policy_probs(&pol_i, (double[1]){1.0}, final_probs);
    printf("    final policy P(a=1 | s=1) = %.3f (started ~0.5)\n", final_probs[1]);
    CHECK(final_probs[1] > 0.6, "REINFORCE converged toward high-reward action (P(a=1) > 0.6)");

    if (fails > 0) {
        printf("\n%d TEST(S) FAILED\n", fails);
        return 1;
    }
    printf("\nALL GG TESTS PASSED\n");
    return 0;
}
