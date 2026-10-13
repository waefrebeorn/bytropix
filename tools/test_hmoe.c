/* test_hmoe.c -- BRAIN #116 hyperbolic MoE routing.
 * Proves: nearest expert by hyperbolic distance, top-k ordering, exact-match
 * routing (token at expert centroid routes there), fail-closed on NULL. */
#include "wubu_hmoe.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_hmoe_t m;
    CHECK(wubu_hmoe_init(&m, -1.0) == 0, "init ok");

    /* three experts at different disk locations */
    double e0[WUBU_HM_DIM] = {0.1, 0, 0, 0, 0, 0, 0, 0};
    double e1[WUBU_HM_DIM] = {-0.2, 0.1, 0, 0, 0, 0, 0, 0};
    double e2[WUBU_HM_DIM] = {0.0, -0.3, 0, 0, 0, 0, 0, 0};
    wubu_hmoe_add_expert(&m, 10, e0);
    wubu_hmoe_add_expert(&m, 11, e1);
    wubu_hmoe_add_expert(&m, 12, e2);

    /* token near e0 -> routes to expert 10 first */
    double tok[WUBU_HM_DIM] = {0.12, 0.01, 0, 0, 0, 0, 0, 0};
    int ids[3]; double dists[3];
    int k = wubu_hmoe_route(&m, tok, 1, ids, dists);
    CHECK(k == 1, "top-1 returns 1");
    CHECK(ids[0] == 10, "token near e0 -> expert 10");

    /* token exactly at e0's centroid -> distance ~0 */
    double exact[WUBU_HM_DIM] = {0.1, 0, 0, 0, 0, 0, 0, 0};
    wubu_hmoe_route(&m, exact, 1, ids, dists);
    CHECK(dists[0] < 1e-6, "token at centroid -> distance ~0");

    /* top-3 ordering: nearest first */
    k = wubu_hmoe_route(&m, tok, 3, ids, dists);
    CHECK(k == 3, "top-3 returns 3");
    CHECK(ids[0] == 10, "nearest is expert 10");
    CHECK(dists[0] <= dists[1] && dists[1] <= dists[2], "distances ascending");
    /* the 3 experts are distinct */
    CHECK(ids[0] != ids[1] && ids[1] != ids[2] && ids[0] != ids[2], "3 distinct experts");

    /* token near e2 -> expert 12 */
    double tok2[WUBU_HM_DIM] = {0.01, -0.28, 0, 0, 0, 0, 0, 0};
    wubu_hmoe_route(&m, tok2, 1, ids, dists);
    CHECK(ids[0] == 12, "token near e2 -> expert 12");

    /* hyperbolic: a point near the boundary is FAR from interior experts even
     * if Euclidean-close. Expert near edge vs token near center. */
    wubu_hmoe_t m2;
    wubu_hmoe_init(&m2, -1.0);
    double deep[WUBU_HM_DIM] = {0.9, 0, 0, 0, 0, 0, 0, 0};   /* near boundary */
    double broad[WUBU_HM_DIM] = {0.1, 0, 0, 0, 0, 0, 0, 0};  /* near center */
    wubu_hmoe_add_expert(&m2, 1, deep);
    wubu_hmoe_add_expert(&m2, 2, broad);
    double tok3[WUBU_HM_DIM] = {0.15, 0, 0, 0, 0, 0, 0, 0};  /* closer to broad */
    wubu_hmoe_route(&m2, tok3, 2, ids, dists);
    CHECK(ids[0] == 2, "hyperbolic routing prefers the near-center expert");

    /* fail-closed */
    CHECK(wubu_hmoe_route(&m, tok, 0, ids, dists) == 0, "k=0 -> 0");
    CHECK(wubu_hmoe_route(NULL, tok, 1, ids, dists) == 0, "NULL route -> 0");
    CHECK(wubu_hmoe_add_expert(NULL, 1, e0) == -1, "NULL add_expert -> -1");
    CHECK(wubu_hmoe_init(NULL, -1) == -1, "NULL init -> -1");

    printf("\n=== test_hmoe: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
