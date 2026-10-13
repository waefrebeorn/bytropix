/* test_compute_eff.c -- PATHS #322 compute-efficiency: architectural wins.
 * Proves: architectural wins (less compute + same/better quality) score
 * highest and are flagged, scaling (more compute) scores low, best technique
 * is the most efficient, fail-closed on NULL/bad mult. */
#include "wubu_compute_eff.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_compute_eff_t c;
    CHECK(wubu_ceff_init(&c) == 0, "init ok");

    /* architectural wins: less compute, same/better quality */
    CHECK(wubu_ceff_add(&c, "linear-attention", 0.3, 0.05) == 0, "add linear-attn (0.3x compute, +quality)");
    CHECK(wubu_ceff_add(&c, "MoE-sparse", 0.5, 0.1) == 0, "add MoE (0.5x compute, +quality)");
    CHECK(wubu_ceff_add(&c, "int8-quant", 0.25, -0.02) == 0, "add int8 (0.25x, tiny quality loss)");
    /* scaling: more compute, more quality (the naive approach) */
    CHECK(wubu_ceff_add(&c, "2x-compute", 2.0, 0.1) == 0, "add 2x-compute scaling");
    CHECK(wubu_ceff_add(&c, "10x-compute", 10.0, 0.15) == 0, "add 10x-compute scaling");

    /* architectural wins flagged */
    CHECK(wubu_ceff_is_win(&c.t[0]) == 1, "linear-attn is an architectural win");
    CHECK(wubu_ceff_is_win(&c.t[1]) == 1, "MoE is an architectural win");
    CHECK(wubu_ceff_is_win(&c.t[3]) == 0, "2x-compute is NOT a win (more compute)");
    /* int8 loses a little quality -> not a strict win, but still efficient */
    CHECK(wubu_ceff_is_win(&c.t[2]) == 0, "int8 (tiny quality loss) not a strict win");

    /* efficiency scores: architectural wins >> scaling */
    double s_lin = wubu_ceff_score(&c.t[0]);   /* 0.05/0.3 = 0.167 */
    double s_scale = wubu_ceff_score(&c.t[3]); /* 0.1/2.0 = 0.05 */
    CHECK(s_lin > s_scale, "linear-attn more efficient than 2x-compute");
    double s_int8 = wubu_ceff_score(&c.t[2]);  /* -0.02/0.25 = -0.08 */
    CHECK(s_lin > 0 && s_scale > 0, "positive-quality techniques have positive score");
    CHECK(s_int8 < 0, "int8 (quality loss) has negative score");

    /* best technique by efficiency */
    int best = wubu_ceff_best(&c);
    CHECK(best >= 0, "best technique found");
    CHECK(best == 1, "MoE is the most efficient (0.1/0.5=0.2 > linear-attn 0.167)");

    /* the architectural-win principle: MoE (0.5x compute, +0.1 quality) beats
     * 10x-compute (10x compute, +0.15 quality) on efficiency */
    double s_moe = wubu_ceff_score(&c.t[1]);   /* 0.1/0.5 = 0.2 */
    double s_10x = wubu_ceff_score(&c.t[4]);   /* 0.15/10 = 0.015 */
    CHECK(s_moe > s_10x * 10, "MoE >> 10x-compute on efficiency (arch wins)");

    /* fail-closed */
    CHECK(wubu_ceff_add(&c, "bad", 0.0, 0.1) == -1, "zero compute_mult -> -1");
    CHECK(wubu_ceff_add(&c, "bad", -1.0, 0.1) == -1, "negative compute_mult -> -1");
    CHECK(wubu_ceff_best(NULL) == -1, "NULL best -> -1");
    CHECK(wubu_ceff_is_win(NULL) == 0, "NULL is_win -> 0");
    CHECK(wubu_ceff_score(NULL) == 0.0, "NULL score -> 0");

    printf("\n=== test_compute_eff: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
