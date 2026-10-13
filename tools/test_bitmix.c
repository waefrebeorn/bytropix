/* test_bitmix.c -- BRAIN #80 mixed per-role bit ladders, never uniform.
 * Proves: per-role widths differ (non-uniform), sensitive roles get more bits
 * than robust, norms/biases full precision, budget respected, >=2 distinct
 * widths always, fail-closed on bad params. */
#include "wubu_bitmix.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_bitmix_t m;
    CHECK(wubu_bitmix_plan(&m, 8, 100) == 0, "plan base=8 budget=100 ok");
    CHECK(wubu_bitmix_plan(&m, 1, 100) == -1, "base<2 rejected");
    CHECK(wubu_bitmix_plan(&m, 8, 5) == -1, "budget too small rejected");
    CHECK(wubu_bitmix_plan(NULL, 8, 100) == -1, "NULL plan -> -1");

    /* re-plan with valid params */
    wubu_bitmix_plan(&m, 8, 100);
    /* NEVER UNIFORM: at least 2 distinct widths */
    CHECK(wubu_bitmix_distinct(&m) >= 2, "non-uniform: >=2 distinct widths");
    /* sensitive (embed) > robust (MoE) */
    CHECK(wubu_bitmix_width(&m, WUBU_ROLE_EMBED) > wubu_bitmix_width(&m, WUBU_ROLE_MOE_EXPERT),
          "embed (sensitive) > MoE (robust)");
    /* norms/biases full precision (16) */
    CHECK(wubu_bitmix_width(&m, WUBU_ROLE_NORM) == 16, "norm full precision (16)");
    CHECK(wubu_bitmix_width(&m, WUBU_ROLE_BIAS) == 16, "bias full precision (16)");
    /* all widths in [2,16] */
    int in_range = 1;
    for (int r = 0; r < WUBU_ROLE_COUNT; r++) {
        int w = wubu_bitmix_width(&m, (wubu_role_t)r);
        if (w < 2 || w > 16) in_range = 0;
    }
    CHECK(in_range == 1, "all widths in [2,16]");
    /* budget respected */
    CHECK(wubu_bitmix_total(&m) <= 100, "total <= budget");

    /* tight budget forces shaving but stays non-uniform + in range */
    wubu_bitmix_t t;
    CHECK(wubu_bitmix_plan(&t, 8, 40) == 0, "tight budget plan ok");
    CHECK(wubu_bitmix_total(&t) <= 40, "tight budget respected");
    CHECK(wubu_bitmix_distinct(&t) >= 2, "still non-uniform under tight budget");
    CHECK(wubu_bitmix_width(&t, WUBU_ROLE_NORM) == 16, "norm stays 16 even tight");

    /* high base width */
    wubu_bitmix_t h;
    wubu_bitmix_plan(&h, 12, 200);
    CHECK(wubu_bitmix_distinct(&h) >= 2, "high base still non-uniform");
    CHECK(wubu_bitmix_width(&h, WUBU_ROLE_MOE_EXPERT) < wubu_bitmix_width(&h, WUBU_ROLE_EMBED),
          "high base: MoE < embed");

    /* fail-closed accessors */
    CHECK(wubu_bitmix_width(NULL, WUBU_ROLE_EMBED) == -1, "NULL width -> -1");
    CHECK(wubu_bitmix_distinct(NULL) == 0, "NULL distinct -> 0");
    CHECK(wubu_bitmix_total(NULL) == 0, "NULL total -> 0");

    printf("\n=== test_bitmix: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
