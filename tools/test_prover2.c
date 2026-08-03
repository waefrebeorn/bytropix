/*
 * test_prover2.c -- the math-RL verifier test (phase 6).
 * The verifier must ACCEPT the true theorems (the Lean-proved ones) and
 * REJECT false steps -- the reward signal has to be trustworthy or the
 * model learns to game it.
 */
#include <stdio.h>
#include "wubu_prover.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

int main(void)
{
    printf("=== test_prover2 (the math-RL verifier) ===\n");
    double c = 0.25;

    /* 1. the Lean-proved theorems must be ACCEPTED */
    CHECK(wubu_prover_mobius_closure(c, 0.3, -0.4) == 1,
          "mobius closure accepted (Lean: mobius_add_preserves_ball)");
    CHECK(wubu_prover_explog(c, 0.5, 1e-9) == 1,
          "exp(log(y))=y accepted (Lean: poincare_exp_log_identity)");
    CHECK(wubu_prover_gyro(c, 0.3, -0.4, 0.5, 1e-9) == 1,
          "gyroassoc accepted (Lean: gyroassoc_1d)");

    /* 2. steps OUTSIDE the ball must be REJECTED */
    CHECK(wubu_prover_mobius_closure(c, 2.5, 0.1) == 0,
          "closure rejected when x outside the ball");
    CHECK(wubu_prover_explog(c, 3.0, 1e-9) == 0,
          "explog rejected when y outside the ball");

    /* 3. the step checker: a valid factor step accepted, an invalid one
     * rejected */
    wubu_pf_step_t good = { WUBU_PF_FACTOR, 0, 0, 0, 42.0, 42.0 };
    wubu_pf_step_t bad = { WUBU_PF_FACTOR, 0, 0, 0, 42.0, 43.0 };
    CHECK(wubu_prover_check(&good) == 1, "factor identity accepted");
    CHECK(wubu_prover_check(&bad) == 0, "wrong identity rejected");

    /* 4. the chain: all-true -> full reward; mixed -> partial */
    wubu_pf_step_t chain[3] = {
        { WUBU_PF_MOBUS, c, 0.3, -0.4, 0, 0 },
        { WUBU_PF_FACTOR, 0, 0, 0, 7.0, 7.0 },
        { WUBU_PF_RING, 5.0, 2.0, 3.0, 0, 0 },   /* 5 = 2+3 */
    };
    CHECK(wubu_prover_check_chain(chain, 3) == 3, "full chain accepted");
    wubu_pf_step_t mixed[3] = {
        { WUBU_PF_MOBUS, c, 0.3, -0.4, 0, 0 },
        { WUBU_PF_FACTOR, 0, 0, 0, 7.0, 8.0 },   /* WRONG */
        { WUBU_PF_RING, 5.0, 2.0, 3.0, 0, 0 },
    };
    CHECK(wubu_prover_check_chain(mixed, 3) == 2, "mixed chain partial reward");
    printf("  chain rewards: full=3, mixed=2 (the RL signal is trustworthy)\n");

    if (failures == 0) printf("ALL PROVER2 TESTS PASSED -- the reward is sound\n");
    else printf("%d PROVER2 FAILURES\n", failures);
    return failures ? 1 : 0;
}
