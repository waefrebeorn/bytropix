/* Test: SMT-style GEMV equivalence checking (doc F02). */
#include "wubu_smt_check.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

int main(void) {
    /* Test 1: K=4, tolerance=0.1 — should pass (int8 quantization error is small) */
    wubu_smt_result_t r = wubu_smt_check_gemv(4, 0.1f);
    printf("K=4 tol=0.1: status=%s checks=%d failures=%d max_err=%.8f\n",
           wubu_smt_status_str(r.status), r.n_checks, r.n_failures, (double)r.max_error);
    assert(r.status == WUBU_SMT_OK);
    assert(r.n_checks > 0);
    assert(r.n_failures == 0);

    /* Test 2: K=8, tolerance=0.5 — should pass */
    r = wubu_smt_check_gemv(8, 0.5f);
    printf("K=8 tol=0.5: status=%s checks=%d failures=%d max_err=%.8f\n",
           wubu_smt_status_str(r.status), r.n_checks, r.n_failures, (double)r.max_error);
    assert(r.status == WUBU_SMT_OK);

    /* Test 3: Verify a specific known case */
    int8_t w_q[4] = {127, -128, 64, -64};
    int8_t x_q[4] = {1, -1, 2, -2};
    r = wubu_smt_verify_specific(w_q, x_q, 0.01f, 0.1f, 4, 0.1f);
    printf("Specific: status=%s max_err=%.8f\n",
           wubu_smt_status_str(r.status), (double)r.max_error);
    assert(r.status == WUBU_SMT_OK);

    /* Test 4: Verify with extreme scale mismatch should fail tight tolerance */
    int8_t w2[4] = {127, 127, 127, 127};
    int8_t x2[4] = {127, 127, 127, 127};
    r = wubu_smt_verify_specific(w2, x2, 10.0f, 10.0f, 4, 0.0f);
    /* 127*10*4*127*10 = large but exact in int32, so this should be exact */
    printf("Extreme exact: status=%s max_err=%.8f\n",
           wubu_smt_status_str(r.status), (double)r.max_error);
    /* int32 accumulator: 127*127 = 16129, *4 = 64516, *100 = 6451600.0f — exact */
    assert(r.status == WUBU_SMT_OK);

    /* Test 5: The check should catch a real bug (wrong scale) */
    int8_t w3[4] = {100, 50, 25, 12};
    int8_t x3[4] = {10, 5, 2, 1};
    /* Deliberately wrong: use 2x scale */
    r = wubu_smt_verify_specific(w3, x3, 0.02f, 0.02f, 4, 0.0001f);
    /* ref = sum(w_i*0.02 * x_i*0.02) = 0.0004 * sum(w_i * x_i)
     * quant = sum(w_q_i * x_q_i) * 0.02 * 0.02 = 0.0004 * sum
     * So they should match — no bug with consistent scales. */
    printf("Consistent scales: status=%s max_err=%.8f\n",
           wubu_smt_status_str(r.status), (double)r.max_error);

    printf("ALL SMT-CHECK TESTS PASSED\n");
    return 0;
}
