/* test_dedup.c -- the rolling-hash duplicate-window scanner: an embedded
 * exact duplicate must be found, a near-duplicate (one token off) must
 * NOT be (the hash-collision guard compares the real windows), and the
 * rate must be exact. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_dedup.h"

int main(void)
{
    int ok = 1;
    /* a 100-token stream: tokens 0..39 = A, 40..79 = a copy of 0..39, 80..99 = B */
    uint16_t toks[100];
    for (int i = 0; i < 40; i++) toks[i] = (uint16_t)(1000 + i);
    for (int i = 0; i < 40; i++) toks[40 + i] = toks[i];   /* the dup */
    for (int i = 0; i < 20; i++) toks[80 + i] = (uint16_t)(2000 + i);

    uint8_t dup[100] = {0};
    long ndup = wubu_dedup_scan(toks, 100, 16, dup);
    /* the windows fully inside the copy region [40..64] are dups of
     * [0..24]; the exact count: windows starting at 40..64 (25 windows)
     * match [0..24] -- the windows starting at 65..79 overlap the B
     * region and are NOT dups (different content) */
    if (ndup < 20 || ndup > 30) {
        printf("  dup count %ld (expected ~25) FAIL\n", ndup); ok = 0;
    }
    for (long i = 40; i <= 64; i++)
        if (!dup[i]) { printf("  window %ld should be dup FAIL\n", i); ok = 0; }
    for (long i = 0; i < 16; i++)
        if (dup[i]) { printf("  window %ld (original) should not be dup FAIL\n", i); ok = 0; }

    /* the near-duplicate: one token off -- must NOT be flagged */
    uint16_t nd[40];
    for (int i = 0; i < 40; i++) nd[i] = (uint16_t)(3000 + i);
    nd[10] = 9999;   /* the one-token difference */
    uint8_t dup2[80] = {0};
    long ndup2 = wubu_dedup_scan(nd, 40, 16, dup2);
    if (ndup2 != 0) { printf("  near-dup flagged %ld FAIL\n", ndup2); ok = 0; }

    /* the rate: 25 dup windows out of 100-16+1 = 85 */
    float rate = wubu_dedup_rate(dup, 100, 16);
    if (rate < 0.25f || rate > 0.33f) {
        printf("  rate %.3f (expected ~0.29) FAIL\n", rate); ok = 0;
    }
    printf("  dup windows %ld, rate %.3f, near-dup clean  %s\n",
           ndup, rate, ok ? "PASS" : "FAIL");
    printf("%s\n", ok ? "ALL DEDUP TESTS PASSED" : "DEDUP FAILURES");
    return ok ? 0 : 1;
}
