/* Test: wubu_cla (Round-3 #223 — Cross-Layer Attention KV sharing). */
#include "wubu_cla.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    /* Mixed attention types: alternating sliding(0)/global(1). */
    int type[10] = {0,0,0,1,1,0,0,1,1,1};
    wubu_cla_t *c = wubu_cla_plan(10, 2, type);
    assert(c);
    /* Type-matched: a global layer shares from the previous global, not a sliding. */
    assert(wubu_cla_kv_owner(c, 0) == 0);   /* run head, self */
    assert(wubu_cla_kv_owner(c, 1) == 0);   /* shares from sliding head 0 */
    assert(wubu_cla_kv_owner(c, 3) == 3);   /* first global -> self */
    assert(wubu_cla_kv_owner(c, 4) == 3);   /* global shares from global 3 */
    assert(wubu_cla_kv_owner(c, 6) == 5);   /* sliding shares from sliding 5 */

    /* Uniform type, share_k=2 => exactly half the layers compute own KV. */
    int uni[10] = {0,0,0,0,0,0,0,0,0,0};
    wubu_cla_t *u = wubu_cla_plan(10, 2, uni);
    double frac = wubu_cla_unique_kv_frac(u);
    printf("CLA uniform unique-KV frac = %.2f (expect 0.50)\n", frac);
    assert(fabs(frac - 0.5) < 1e-9);
    double red = wubu_cla_kv_reduction(u, 100.0);
    printf("CLA KV reduction = %.2f (expect 0.50)\n", red);
    assert(fabs(red - 0.5) < 1e-9);

    wubu_cla_free(c); wubu_cla_free(u);
    assert(wubu_cla_plan(0, 2, type) == NULL);  /* DA: bad args */
    printf("ALL CLA TESTS PASSED\n");
    return 0;
}
