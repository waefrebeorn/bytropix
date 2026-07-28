/* Test: wubu_mhc (Round-3 #213 — mHC identity + non-negativity). */
#include "wubu_mhc.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main(void) {
    wubu_mhc_t *m = wubu_mhc_create(8, 4);
    assert(m);
    /* Identity check at init (manifold constraint). */
    int ok = wubu_mhc_identity_ok(m);
    printf("mHC identity_ok (init) = %d (expect 1)\n", ok);
    assert(ok);
    /* Forward on a basis vector must be finite and bounded (sigmoid non-neg). */
    float x[8]; for (int i=0;i<8;i++) x[i] = (i==3)?1.0f:0.0f;
    float y[8]; wubu_mhc_forward(m, x, NULL, y);
    for (int i=0;i<8;i++) { assert(isfinite(y[i])); assert(y[i] >= -1e-6f && y[i] <= 1.0f+1e-6f); }
    printf("mHC forward bounded in [0,1] on init weights: OK\n");
    /* Exact-identity reconstruction via public API: forward must pass x through. */
    wubu_mhc_set_identity(m);
    wubu_mhc_forward(m, x, NULL, y);
    printf("mHC exact-identity forward[3] = %.4f (expect ~1)\n", y[3]);
    assert(fabsf(y[3] - 1.0f) < 1e-2f);
    /* Non-negativity constraint. */
    float w[4] = {-5.0f, 0.0f, 5.0f, 100.0f};
    wubu_mhc_apply_nonneg(w, 4);
    for (int i=0;i<4;i++) { assert(w[i] >= 0.0f && w[i] <= 1.0f); }
    printf("mHC sigmoid non-neg in [0,1]: OK\n");
    wubu_mhc_free(m);
    /* NULL guard. */
    assert(wubu_mhc_create(0, 4) == NULL);
    assert(wubu_mhc_identity_ok(NULL) == 0);
    printf("ALL MHC TESTS PASSED\n");
    return 0;
}
