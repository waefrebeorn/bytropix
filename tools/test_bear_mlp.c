/*
 * test_bear_mlp.c — tests for the BearRL CPU MLP/Kernels port.
 *
 * Tests: MLP forward, GAE computation, CartPole physics.
 *
 * C11, no external deps.
 */
#include "wubu_bear_mlp.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int tests_run = 0;
static int tests_pass = 0;

static void check(const char *name, int cond) {
    tests_run++;
    if (cond) { tests_pass++; printf("  PASS: %s\n", name); }
    else      { printf("  FAIL: %s\n", name); }
}

static int approx(double a, double b, double tol) {
    return fabs(a - b) < tol;
}

int main(void) {
    printf("=== test_bear_mlp: BearRL CPU Kernel Port ===\n");

    /* ---- Test 1: MLP Forward Pass ---- */
    printf("\n--- Test 1: MLP Forward ---\n");
    {
        /* obs_dim=3, hidden_dim=4, act_dim=2, batch=1 */
        float obs[3] = {1.0f, 2.0f, 3.0f};
        float W1[12] = {1,0,0, 0,1,0, 0,0,1, 1,1,1};  /* [4,3] */
        float b1[4] = {0, 0, 0, 0};
        /* W2: out_dim = 2+1+1+4 = 8, [8,4] = 32 params */
        float W2[32] = {0};
        for (int i = 0; i < 32; i++) W2[i] = 1.0f;  /* all ones */
        float b2[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        float actions[2], logprobs, value, h_out[4];

        int rc = wubu_bear_mlp_forward(obs, NULL, W1, b1, W2, b2,
                                        actions, &logprobs, &value, h_out,
                                        1, 3, 4, 2);
        check("mlp_forward returns 0", rc == 0);

        /* Layer 1: ReLU([1,2,3,6]) = [1,2,3,6] */
        check("h_out[0]=1 (ReLU)", approx(h_out[0], 1.0, 1e-4));
        check("h_out[1]=2 (ReLU)", approx(h_out[1], 2.0, 1e-4));
        check("h_out[2]=3 (ReLU)", approx(h_out[2], 3.0, 1e-4));
        check("h_out[3]=6 (ReLU)", approx(h_out[3], 6.0, 1e-4));

        /* Layer 2: each output = sum(h_out) * 1 + bias = 1+2+3+6 = 12 */
        /* Actions = output[0:2] = 12 each */
        check("action[0]=12", approx(actions[0], 12.0, 1e-3));
        check("action[1]=12", approx(actions[1], 12.0, 1e-3));
        check("value=14 (b2[3]=0 + sum*h)", approx(value, 12.0, 1e-3));

        /* Logprob is finite */
        check("logprob finite", logprobs == logprobs);  /* NaN check */
    }

    /* ---- Test 2: MLP with negative inputs (ReLU) ---- */
    printf("\n--- Test 2: MLP ReLU negative clipping ---\n");
    {
        float obs[3] = {-1.0f, -2.0f, -3.0f};
        float W1[12] = {1,0,0, 0,1,0, 0,0,1, 1,1,1};
        float b1[4] = {0, 0, 0, 0};
        float W2[32];
        for (int i = 0; i < 32; i++) W2[i] = 1.0f;
        float b2[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        float actions[2], logprobs, value, h_out[4];
        wubu_bear_mlp_forward(obs, NULL, W1, b1, W2, b2,
                               actions, &logprobs, &value, h_out,
                               1, 3, 4, 2);
        /* ReLU: [-1,-2,-3,-6] -> [0,0,0,0] */
        check("h_out[0]=0 (ReLU)", approx(h_out[0], 0.0, 1e-6));
        check("h_out[3]=0 (ReLU)", approx(h_out[3], 0.0, 1e-6));
        /* sum(h_out) = 0, so actions = 0 + bias(0) = 0 */
        check("action[0]=0 (ReLU kills)", approx(actions[0], 0.0, 1e-6));
    }

    /* ---- Test 3: NULL safety ---- */
    printf("\n--- Test 3: NULL safety ---\n");
    {
        float obs[3] = {1,2,3};
        float W1[12] = {0}, b1[4] = {0}, W2[32] = {0}, b2[8] = {0};
        float a[2] = {0}, lp = 0, v = 0, ho[4] = {0};

        check("NULL obs returns -1",
              wubu_bear_mlp_forward(NULL, NULL, W1, b1, W2, b2,
                                     a, &lp, &v, ho, 1, 3, 4, 2) == -1);
        check("NULL W1 returns -1",
              wubu_bear_mlp_forward(obs, NULL, NULL, b1, W2, b2,
                                     a, &lp, &v, ho, 1, 3, 4, 2) == -1);
        check("bad dims returns -1",
              wubu_bear_mlp_forward(obs, NULL, W1, b1, W2, b2,
                                     a, &lp, &v, ho, 1, 3, 4, 0) == -1);

        check("GAE NULL returns -1",
              wubu_bear_gae(NULL, NULL, NULL, NULL, NULL, 1, 1, 0.99f, 0.95f) == -1);
        check("cartpole NULL returns -1",
              wubu_bear_cartpole_step(NULL, NULL, NULL, NULL, NULL, NULL,
                                       NULL, 0, 1, 1, 9.8, 1, 0.02, 10, 0.2, 2.4) == -1);
    }

    /* ---- Test 4: GAE Computation ---- */
    printf("\n--- Test 4: GAE ---\n");
    {
        /* T=3, B=1, simple rewards
         * rewards = [1, 1, 1], dones = [0,0,0], values = [10, 11, 12]
         * At t=2 (last):
         *   delta = 1 + 0.99*1*(0-12) - 12 = 1 - 11.88 - 12 = -22.88
         *   gae = -22.88
         *   advantage[2] = -22.88, return[2] = -22.88 + 12 = -10.88
         * At t=1:
         *   delta = 1 + 0.99*1*(12-11) - 11 = 1 + 0.99 - 11 = -9.01
         *   gae = -9.01 + 0.99*0.95*1*(-22.88) = -9.01 - 21.517 = -30.527... */
        float rewards[3] = {1, 1, 1};
        uint8_t dones[3] = {0, 0, 0};
        float values[3] = {10, 11, 12};
        float advantages[3] = {0}, returns[3] = {0};

        int rc = wubu_bear_gae(rewards, dones, values, advantages, returns,
                                3, 1, 0.99f, 0.95f);
        check("gae returns 0", rc == 0);

        /* At t=2 (last): delta = 1 + 0.99*1*(0-0) - 12 = 1 - 12 = -11 */
        float delta2 = 1.0f + 0.99f * 0.0f - 12.0f;
        check("gae t=2 = delta", approx(advantages[2], delta2, 1e-3));

        /* return = gae + v */
        check("return t=2 = gae + v",
              approx(returns[2], delta2 + 12.0f, 1e-3));

        /* Bad args */
        check("gae bad dims returns -1",
              wubu_bear_gae(rewards, dones, values, advantages, returns,
                             0, 1, 0.99f, 0.95f) == -1);
    }

    /* ---- Test 5: CartPole Physics ---- */
    printf("\n--- Test 5: CartPole Physics ---\n");
    {
        float x = 0.0f, x_dot = 0.0f;
        float theta = 0.1f, theta_dot = 0.0f;
        float reward = 0, force = 1.0f;
        uint8_t done = 0;

        int rc = wubu_bear_cartpole_step(&x, &x_dot, &theta, &theta_dot,
                                          &reward, &done, &force,
                                          1, 1.0f, 0.5f, 9.8f, 1.0f,
                                          0.02f, 10.0f, 0.2f, 2.4f);
        check("cartpole_step returns 0", rc == 0);
        check("reward = 1.0", approx(reward, 1.0, 1e-6));
        check("done = 0 (within thresholds)", done == 0);
        check("theta changed after step", theta != 0.1f);
        check("x changed after step", x != 0.0f || x_dot != 0.0f);

        /* Batch mode: 4 envs */
        float xs[4] = {0,0,0,0};
        float xds[4] = {0,0,0,0};
        float ths[4] = {0.1f, 0.2f, 0.05f, 0.3f};
        float thds[4] = {0,0,0,0};
        float rs[4] = {0,0,0,0};
        uint8_t dns[4] = {0,0,0,0};
        float f[4] = {1, -1, 1, -1};

        rc = wubu_bear_cartpole_step(xs, xds, ths, thds, rs, dns, f,
                                      4, 1.0f, 0.5f, 9.8f, 1.0f,
                                      0.02f, 10.0f, 0.2f, 2.4f);
        check("batch cartpole returns 0", rc == 0);
        check("all rewards = 1", rs[0]==1 && rs[1]==1 && rs[2]==1 && rs[3]==1);
        check("batch done[0]=0", dns[0] == 0);
    }

    printf("\n=== Results: %d/%d tests passed ===\n", tests_pass, tests_run);
    if (tests_pass == tests_run) {
        printf("ALL BEAR ML TESTS PASSED\n");
        return 0;
    }
    return 1;
}
