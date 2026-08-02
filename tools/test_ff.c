/*
 * test_ff.c -- FF01-FF07 verification.
 */
#include "wubu_gp.h"
#include "wubu_acq.h"
#include "wubu_bo.h"
#include "wubu_uq.h"
#include "wubu_active.h"
#include "wubu_bandit.h"
#include <stdio.h>
#include <math.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fails++; printf("FAIL: %s\n", ""); printf("    -> %s\n", msg); } \
    else printf("  ok: %s\n", msg); \
} while(0)

/* Toy black-box objective: tok_s as function of 2-dim config.
   f(x) = 25 + 5*sin(x0) - 3*x1^2 (has a clear optimum) */
static double toy_obj(const double *x, int dim, void *ctx) {
    (void)ctx;
    if (dim < 2) return 0;
    return 25.0 + 5.0 * sin(x[0]) - 3.0 * x[1] * x[1];
}

int main() {
    /* FF01: Gaussian Process surrogate */
    printf("=== FF01: Gaussian Process ===\n");
    wubu_gp_t gp;
    memset(&gp, 0, sizeof(gp));
    gp.dim = 2;
    gp.sigma2_f = 1.0;
    gp.length_scale = 1.0;
    gp.noise = 1e-3;
    /* Seed with a few observations */
    double x0[2] = {0.0, 0.0}, x1[2] = {1.0, 0.5}, x2[2] = {-1.0, 0.3};
    wubu_gp_add(&gp, x0, toy_obj(x0, 2, NULL));
    wubu_gp_add(&gp, x1, toy_obj(x1, 2, NULL));
    wubu_gp_add(&gp, x2, toy_obj(x2, 2, NULL));
    CHECK(wubu_gp_fit(&gp) == 0, "gp fit (Cholesky)");
    double mean, var;
    CHECK(wubu_gp_predict(&gp, x0, &mean, &var) == 0, "gp predict at observed point");
    CHECK(fabs(mean - toy_obj(x0, 2, NULL)) < 1.0, "gp mean ~ observed value at training point");
    CHECK(var >= 0, "gp variance non-negative");
    /* At a far point, variance should be higher */
    double xfar[2] = {5.0, 5.0};
    double mfar, vfar;
    wubu_gp_predict(&gp, xfar, &mfar, &vfar);
    printf("    var(near)=%.4f var(far)=%.4f\n", var, vfar);
    CHECK(vfar > var, "gp variance higher at far/unobserved point");

    /* FF02: Acquisition functions */
    printf("\n=== FF02: Acquisition Functions ===\n");
    wubu_acq_t acq_ei = { WUBU_ACQ_EI, 20.0, 0.0, 0.01 };
    wubu_acq_t acq_ucb = { WUBU_ACQ_UCB, 20.0, 2.0, 0.0 };
    wubu_acq_t acq_pi = { WUBU_ACQ_PI, 20.0, 0.0, 0.0 };
    /* High mean, moderate std → all positive */
    double ei = wubu_acq_value(&acq_ei, 30.0, 2.0);
    double ucb = wubu_acq_value(&acq_ucb, 30.0, 2.0);
    double pi = wubu_acq_value(&acq_pi, 30.0, 2.0);
    CHECK(ei > 0, "EI positive at promising point (mean >> f*)");
    CHECK(ucb > 30.0, "UCB = mean + kappa*std > mean");
    CHECK(pi > 0.9, "PI near 1.0 when mean >> f*");
    /* Low mean → EI should be lower than at high mean for same std */
    double ei_low = wubu_acq_value(&acq_ei, 10.0, 2.0);
    CHECK(ei_low < ei, "EI lower when mean is below incumbent");

    /* FF03: Bayesian Optimization loop */
    printf("\n=== FF03: Bayesian Optimization ===\n");
    wubu_bo_t bo;
    memset(&bo, 0, sizeof(bo));
    bo.dim = 2;
    bo.n_cand = 12;
    /* Candidate grid */
    int ci = 0;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++) {
            bo.cand[ci][0] = (i - 1.5) * 0.5;
            bo.cand[ci][1] = (j - 1.0) * 0.4;
            ci++;
        }
    wubu_acq_t acq = { WUBU_ACQ_UCB, 20.0, 2.0, 0.0 };
    /* Run a few BO steps */
    for (int step = 0; step < 5; step++) {
        wubu_bo_step(&gp, &acq, &bo, toy_obj, NULL);
    }
    CHECK(gp.n >= 8, "BO added 5 more observations (total >= 8)");
    CHECK(bo.best_acq >= 0, "BO selected a candidate");

    /* FF04: Uncertainty Quantification */
    printf("\n=== FF04: Uncertainty Quantification ===\n");
    wubu_uq_t uq;
    memset(&uq, 0, sizeof(uq));
    double b0[5] = {10, 12, 11, 13, 10};
    double b1[5] = {10.5, 11.5, 12, 12.5, 10.5};
    double b2[5] = {9.5, 12.5, 10.5, 13.5, 9.5};
    wubu_uq_add_boot(&uq, b0, 5);
    wubu_uq_add_boot(&uq, b1, 5);
    wubu_uq_add_boot(&uq, b2, 5);
    CHECK(wubu_uq_fit(&uq) == 0, "uq fit (bootstrap variance)");
    CHECK(fabs(uq.mean[0] - 10.0) < 1e-6, "uq mean[0] = 10.0 (avg of bootstraps)");
    CHECK(uq.var[0] > 0, "uq variance[0] > 0 (bootstrap spread)");
    double res[5] = {0.5, -0.3, 0.2, -0.4, 0.1};
    wubu_uq_calibrate(&uq, res, 5, 0.2);
    double lo, hi;
    wubu_uq_interval(&uq, 0, &lo, &hi);
    printf("    interval[0]: [%.3f, %.3f]\n", lo, hi);
    CHECK(lo < uq.mean[0] && hi > uq.mean[0], "uq interval contains mean");

    /* FF05: Active Learning */
    printf("\n=== FF05: Active Learning ===\n");
    wubu_active_t al;
    memset(&al, 0, sizeof(al));
    al.n = 5;
    double av[5] = {0.1, 0.9, 0.3, 0.7, 0.2};  /* variances */
    int adis[5] = {1, 5, 2, 4, 1};             /* committee disagreement */
    for (int i = 0; i < 5; i++) al.var[i] = av[i];
    for (int i = 0; i < 5; i++) al.committee_disagree[i] = adis[i];
    int ui;
    CHECK(wubu_active_uncertainty(&al, &ui) == 0, "uncertainty sampling");
    CHECK(ui == 1, "uncertainty picks highest-var point (idx 1, var 0.9)");
    wubu_active_query(&al, 1);
    CHECK(wubu_active_uncertainty(&al, &ui) == 0 && ui == 3, "after query, picks next highest (idx 3)");
    int qi;
    CHECK(wubu_active_qbc(&al, &qi) == 0 && qi == 3, "QBC picks next highest-disagreement (idx 3, disagree 4; idx 1 already queried)");

    /* FF06: Thompson Sampling / Bandits */
    printf("\n=== FF06: Thompson Sampling ===\n");
    wubu_bandit_t bandit;
    CHECK(wubu_bandit_init(&bandit, 4) == 0, "bandit init (4 arms, Beta(1,1))");
    unsigned seed = 42;
    int arm = wubu_bandit_sample(&bandit, &seed);
    CHECK(arm >= 0 && arm < 4, "bandit sample returns valid arm");
    /* Make arm 0 always reward → it should dominate over time */
    for (int t = 0; t < 50; t++) {
        int a = wubu_bandit_sample(&bandit, &seed);
        wubu_bandit_update(&bandit, a, (a == 0) ? 1 : 0);
    }
    CHECK(bandit.rewards[0] > 0, "arm 0 accumulated rewards");
    CHECK(bandit.pulls[0] >= bandit.pulls[1], "arm 0 pulled at least as much as loser arm 1");

    /* FF07: Integration — BO converges to optimum with UQ */
    printf("\n=== FF07: Integration ===\n");
    /* After BO, best observed y should be near toy optimum (max is at x0=π/2, x1=0) */
    double best_y = -1e9;
    for (int i = 0; i < gp.n; i++) if (gp.y[i] > best_y) best_y = gp.y[i];
    CHECK(best_y > 27.0, "BO found tok_s > 27 (near true optimum ~30)");
    printf("    best observed tok_s = %.3f\n", best_y);

    /* Summary */
    if (fails > 0) {
        printf("\n%d TEST(S) FAILED\n", fails);
        return 1;
    }
    printf("\nALL FF TESTS PASSED\n");
    return 0;
}
