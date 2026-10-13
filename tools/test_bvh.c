/* test_bvh.c -- WORLDSIM #269 BVH acceleration structure.
 * Proves: build a tree, query returns correct overlap count, pruning (query
 * in empty region = 0), balanced depth ~log(n), leaf bodies counted, fail-closed. */
#include "wubu_bvh.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_bvh_t t;
    CHECK(wubu_bvh_init(&t) == 0, "init ok");

    /* 16 bodies in a 4x4 grid, radius 0.5, spaced 10 apart */
    for (int i = 0; i < 16; i++) {
        int gx = i % 4, gy = i / 4;
        CHECK(wubu_bvh_add(&t, gx * 10.0, gy * 10.0, 0.5, i) >= 0, "add body");
    }
    CHECK(t.n_bodies == 16, "16 bodies added");

    /* build */
    int nn = wubu_bvh_build(&t);
    CHECK(nn > 0, "build produced nodes");
    CHECK(t.n_nodes >= 16, "at least one node per body (leaves)");
    CHECK(wubu_bvh_depth(&t) > 1, "tree has depth > 1");

    /* query around body 0 (at 0,0): should find exactly 1 (only body 0 in that cell) */
    int q0 = wubu_bvh_query(&t, -1, -1, 1, 1);
    CHECK(q0 == 1, "query near body 0 finds 1");

    /* query a big box covering everything: finds all 16 */
    int qall = wubu_bvh_query(&t, -10, -10, 40, 40);
    CHECK(qall == 16, "big query finds all 16");

    /* query in empty region (between grid points, far away): 0 */
    int qempty = wubu_bvh_query(&t, 100, 100, 110, 110);
    CHECK(qempty == 0, "empty-region query finds 0 (pruned)");

    /* query covering 2 adjacent bodies (0 at 0,0 and 1 at 10,0) */
    int q2 = wubu_bvh_query(&t, -2, -2, 12, 2);
    CHECK(q2 == 2, "query spanning bodies 0,1 finds 2");

    /* balanced depth: 16 leaves -> depth ~ log2(16)+1 = 5 (allow range) */
    int depth = wubu_bvh_depth(&t);
    CHECK(depth >= 4 && depth <= 6, "balanced depth ~log2(n) (4..6 for 16)");

    /* fail-closed */
    CHECK(wubu_bvh_query(NULL, 0, 0, 1, 1) == 0, "NULL query -> 0");
    CHECK(wubu_bvh_build(NULL) == -1, "NULL build -> -1");
    CHECK(wubu_bvh_add(NULL, 0, 0, 1, 0) == -1, "NULL add -> -1");
    CHECK(wubu_bvh_depth(NULL) == 0, "NULL depth -> 0");

    /* empty BVH */
    wubu_bvh_t e;
    wubu_bvh_init(&e);
    CHECK(wubu_bvh_build(&e) == -1, "empty build -> -1");

    printf("\n=== test_bvh: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
