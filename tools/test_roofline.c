/* Test: wubu_roofline (Round-2 #101 — B*-crossover auto-tuner). */
#include "wubu_roofline.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

int main(void) {
    wubu_roofline_cfg_t c = wubu_roofline_default();
    double P = 70.6e9;  /* Llama-3 70B */
    /* FP16 weights + FP16 KV, s=4096: B* ~ 108 (survey). */
    double bstar = wubu_roofline_bstar(&c, P, 4096);
    printf("B* (FP16/W,FP16/KV,4k) = %.1f (expect ~105-108)\n", bstar);
    assert(fabs(bstar - 108.0) < 5.0);

    /* At B=8 < B* -> compress WEIGHTS. */
    assert(wubu_roofline_advise(&c, P, 8, 4096) == WUBU_COMPRESS_WEIGHTS);
    /* At B=128 > B* -> compress KV. */
    assert(wubu_roofline_advise(&c, P, 128, 4096) == WUBU_COMPRESS_KV);

    /* INT4 weights pulls B* down ~4x -> ~27. */
    c.bw_bits = 4;
    double bstar_i4 = wubu_roofline_bstar(&c, P, 4096);
    printf("B* (INT4/W,FP16/KV,4k) = %.1f (expect ~27)\n", bstar_i4);
    assert(fabs(bstar_i4 - 27.0) < 5.0);
    c.bw_bits = 16; /* restore */

    /* TPOT sanity: FP16/FP16 at B=32,s=4096 -> ~68 ms (survey). */
    double tpot = wubu_roofline_tpot_ms(&c, P, 32, 4096);
    printf("TPOT(B=32,s=4k) = %.1f ms (expect ~68)\n", tpot);
    assert(fabs(tpot - 68.0) < 8.0);

    /* DA: zero-bandwidth config must not divide-by-zero / produce inf. */
    wubu_roofline_cfg_t cz = c; cz.beta_eff_tb_s = 0;
    double tpot0 = wubu_roofline_tpot_ms(&cz, P, 32, 4096);
    printf("TPOT(zero-BW) = %.1f (expect 0, no inf/nan)\n", tpot0);
    assert(tpot0 == 0.0 && !isinf(tpot0) && !isnan(tpot0));

    printf("ALL ROOFLINE TESTS PASSED\n");
    return 0;
}
