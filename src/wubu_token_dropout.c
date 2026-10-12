/* wubu_token_dropout.c — token dropout (BRAIN #240).
 * C11, self-contained. Deterministic xorshift64* RNG; each token dropped to
 * mask_id with probability p. Returns the dropped count.
 */
#include "wubu_token_dropout.h"

void wubu_rng_seed(wubu_rng_t *r, uint64_t s) {
    if (s == 0) s = 0x9E3779B97F4A7C15ULL;   /* xorshift needs nonzero state */
    r->s = s;
}

double wubu_rng_next(wubu_rng_t *r) {
    /* xorshift64* */
    uint64_t x = r->s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->s = x;
    uint64_t out = x * 0x2545F4914F6CDD1DULL;
    /* Map top 53 bits to [0,1). */
    return (double)(out >> 11) * (1.0 / 9007199254740992.0);
}

int wubu_token_dropout(int *tokens, int n, double p, int mask_id, wubu_rng_t *rng) {
    if (!tokens || n <= 0 || !rng) return -1;
    if (p < 0.0 || p > 1.0) return -1;
    int dropped = 0;
    for (int i = 0; i < n; i++) {
        if (wubu_rng_next(rng) < p) { tokens[i] = mask_id; dropped++; }
    }
    return dropped;
}
