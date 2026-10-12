/* test_token_dropout.c — BRAIN #240: token dropout forward verified.
 *
 * Measured assertions on wubu_token_dropout (+ deterministic RNG):
 *   1. p=0 drops nothing; p=1 drops everything.
 *   2. drop rate ~ p over many tokens (statistical, within tolerance).
 *   3. dropped tokens become exactly mask_id; kept tokens unchanged.
 *   4. deterministic: same seed -> same result (reproducible).
 *   5. returned count == number of mask_id in the array.
 *   6. bad args rejected.
 */
#include "wubu_token_dropout.h"
#include <stdio.h>
#include <math.h>

static int g_run = 0, g_pass = 0;
#define T(cond, msg) do { \
    g_run++; \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while (0)

int main(void) {
    printf("=== test_token_dropout (token dropout, #240) ===\n");
    const int N = 20000, MASK = -1;
    const double p = 0.3;

    /* 1: p=0 and p=1 extremes. */
    int t0[100]; for (int i=0;i<100;i++) t0[i]=i;
    wubu_rng_t r; wubu_rng_seed(&r, 42);
    T(wubu_token_dropout(t0, 100, 0.0, MASK, &r) == 0, "p=0 drops nothing");
    int kept = 1; for (int i=0;i<100;i++) if (t0[i]!=i) kept = 0;
    T(kept, "p=0 leaves all tokens unchanged");
    wubu_rng_seed(&r, 42);
    T(wubu_token_dropout(t0, 100, 1.0, MASK, &r) == 100, "p=1 drops all");

    /* 2: rate ~ p over N tokens. */
    int t[20000]; for (int i=0;i<N;i++) t[i]=i+1;   /* ids >=1, MASK=-1 distinct */
    wubu_rng_seed(&r, 12345);
    int dropped = wubu_token_dropout(t, N, p, MASK, &r);
    double rate = (double)dropped / (double)N;
    T(fabs(rate - p) < 0.02, "drop rate ~ p (within 0.02)");

    /* 3: dropped==mask, kept==original id. */
    int mask_ok = 1;
    for (int i = 0; i < N; i++) {
        if (t[i] == MASK) continue;
        if (t[i] != i + 1) mask_ok = 0;   /* kept token must equal its id */
    }
    T(mask_ok, "kept tokens unchanged, dropped == mask_id");

    /* 4: deterministic (same seed -> same drop count). */
    int t2[20000]; for (int i=0;i<N;i++) t2[i]=i+1;
    wubu_rng_seed(&r, 12345);
    int dropped2 = wubu_token_dropout(t2, N, p, MASK, &r);
    T(dropped == dropped2, "same seed -> same drop count");

    /* 5: returned count == actual mask_id count. */
    int actual = 0; for (int i=0;i<N;i++) if (t[i]==MASK) actual++;
    T(dropped == actual, "returned count == mask_id count");

    /* 6: bad args. */
    T(wubu_token_dropout(NULL, N, p, MASK, &r) == -1, "NULL rejected");
    T(wubu_token_dropout(t, 0, p, MASK, &r) == -1, "n<=0 rejected");
    T(wubu_token_dropout(t, N, 1.5, MASK, &r) == -1, "p>1 rejected");
    T(wubu_token_dropout(t, N, -0.1, MASK, &r) == -1, "p<0 rejected");
    T(wubu_token_dropout(t, N, p, MASK, NULL) == -1, "NULL rng rejected");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}
