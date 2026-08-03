/* test_traj_grpo.c -- the trajectory-level GRPO against the FD oracle.
 * The gradients MUST match the finite differences of the loss (the DA
 * doctrine: tests != correct -- the FD is the source of truth). Also
 * checks the masking (obs tokens get zero gradient) and the no-1/T
 * normalization (longer tasks are not down-weighted). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_traj_grpo.h"

#define G 3
#define T 8

static int run_case(float clip_lo, float clip_hi, int use_old, float *maxrel)
{
    float logp[G * T], mask[G * T], r[G], old[G * T];
    srand(20260804);
    for (int i = 0; i < G * T; i++) {
        logp[i] = -((float)(rand() % 1000)) / 100.0f;   /* negative log-probs */
        old[i] = logp[i] + ((float)(rand() % 200) - 100) / 1000.0f;
    }
    for (int g = 0; g < G; g++)
        for (int t = 0; t < T; t++)
            mask[g * T + t] = (t % 2 == 0) ? 1.0f : 0.0f; /* even = assistant */
    r[0] = 1.0f; r[1] = -0.5f; r[2] = 2.0f;
    float eps = 1e-4f;
    float loss = 0, grad[G * T];
    int ok = wubu_traj_grpo(logp, mask, r, G, T, clip_lo, clip_hi,
                            use_old ? old : NULL, eps, &loss, grad);
    if (!ok) return 0;
    /* the FD check on the masked tokens */
    double maxr = 0;
    float epsf = 1e-3f;
    for (int i = 0; i < G * T; i++) {
        if (mask[i] <= 0) {
            if (fabsf(grad[i]) > 1e-6f) return 0; /* masked -> zero grad */
            continue;
        }
        float save = logp[i];
        logp[i] = save + epsf;
        float l1; wubu_traj_grpo(logp, mask, r, G, T, clip_lo, clip_hi,
                                 use_old ? old : NULL, eps, &l1, NULL);
        logp[i] = save - epsf;
        float l2; wubu_traj_grpo(logp, mask, r, G, T, clip_lo, clip_hi,
                                 use_old ? old : NULL, eps, &l2, NULL);
        logp[i] = save;
        double fd = (l1 - l2) / (2.0 * epsf);
        double rel = fabs(fd - grad[i]) / (fabs(fd) + 1e-9);
        if (rel > maxr) maxr = rel;
    }
    *maxrel = (float)maxr;
    if (maxr >= 5e-2) {
        for (int i = 0; i < G * T; i++) {
            if (mask[i] <= 0) continue;
            float save = logp[i];
            logp[i] = save + 1e-3f;
            float l1; wubu_traj_grpo(logp, mask, r, G, T, clip_lo, clip_hi,
                                     use_old ? old : NULL, eps, &l1, NULL);
            logp[i] = save - 1e-3f;
            float l2; wubu_traj_grpo(logp, mask, r, G, T, clip_lo, clip_hi,
                                     use_old ? old : NULL, eps, &l2, NULL);
            logp[i] = save;
            double fd = (l1 - l2) / 2e-3;
            double rel = fabs(fd - grad[i]) / (fabs(fd) + 1e-9);
            if (rel > 1e-2) {
                printf("    elem %d: lp=%.4f old=%.4f ratio=%.4f fd=%.6f grad=%.6f rel=%.3f\n",
                       i, logp[i], old[i], expf(logp[i] - old[i]), fd, grad[i], rel);
            }
        }
    }
    return maxr < 5e-2;
}

int main(void)
{
    float mr1 = 0, mr2 = 0;
    int a = run_case(0, 0, 0, &mr1);            /* plain advantage NLL */
    int b = run_case(0.2f, 0.28f, 1, &mr2);     /* asymmetric PPO clip */
    printf("  plain-NLL  FD maxrel %.3e  %s\n", mr1, a ? "PASS" : "FAIL");
    printf("  ppo-clip   FD maxrel %.3e  %s\n", mr2, b ? "PASS" : "FAIL");
    /* the no-1/T check: a longer trajectory (more masked tokens) gets the
     * SAME per-token gradient as a short one at equal reward -- the
     * normalization is over the masked count, not the length */
    float logp2[2 * 16], mask2[2 * 16], r2[2];
    for (int i = 0; i < 32; i++) { logp2[i] = -0.5f; mask2[i] = 0; }
    for (int t = 0; t < 8; t++) mask2[t] = 1;      /* traj 0: 8 masked */
    for (int t = 0; t < 16; t++) mask2[16 + t] = 1; /* traj 1: 16 masked */
    r2[0] = 1.0f; r2[1] = 1.0f;                     /* equal rewards */
    float loss2, grad2[32];
    wubu_traj_grpo(logp2, mask2, r2, 2, 16, 0, 0, NULL, 1e-4f, &loss2, grad2);
    /* equal rewards -> equal advantage -> the same per-token grad even
     * though traj 1 is twice as long (the no-1/T doctrine) */
    float g0 = grad2[0], g1 = grad2[16];
    int c = (fabsf(g0 - g1) < 1e-6f);
    printf("  no-1/T:   short %.6f long %.6f %s\n", g0, g1, c ? "PASS" : "FAIL");
    printf("%s\n", (a && b && c) ? "ALL TRAJ-GRPO TESTS PASSED"
                                 : "TRAJ-GRPO FAILURES");
    return (a && b && c) ? 0 : 1;
}
