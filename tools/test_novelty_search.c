/* test_novelty_search.c -- AGENT #164 tests: novelty-search anti-local-optimum. */
#include "wubu_novelty_search.h"
#include <stdio.h>
#include <math.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

static void D(float *d, float a0,float a1,float a2,float a3,float a4,float a5,float a6,float a7){
    d[0]=a0;d[1]=a1;d[2]=a2;d[3]=a3;d[4]=a4;d[5]=a5;d[6]=a6;d[7]=a7;
}

int main(void)
{
    wubu_novelty_search_t ns;
    float desc[WUBU_NS_DIM], nov;

    /* init: empty archive, k=15, threshold=5.0 */
    CHECK(wubu_ns_init(&ns, 5.0f) == 0);
    CHECK(wubu_ns_archive_size(&ns) == 0);
    CHECK(ns.k == WUBU_NS_K);

    /* empty archive -> novelty 0 */
    D(desc, 1,0,0,0,0,0,0,0);
    CHECK(wubu_ns_novelty(&ns, desc, &nov) == 0);
    CHECK(nov == 0.0f);

    /* evaluate on empty archive: novelty 0 < threshold -> rejected */
    CHECK(wubu_ns_evaluate(&ns, desc, 10) == 0);
    CHECK(ns.n_rejected == 1);
    CHECK(ns.n_archived == 0);
    CHECK(wubu_ns_archive_size(&ns) == 0);

    /* seed one behavior at origin */
    float origin[WUBU_NS_DIM] = {0,0,0,0,0,0,0,0};
    CHECK(wubu_ns_seed(&ns, origin, 5) == 0);
    CHECK(wubu_ns_archive_size(&ns) == 1);
    CHECK(ns.n_archived == 1);

    /* a behavior very close to origin: novelty tiny -> rejected */
    float near[WUBU_NS_DIM] = {0.1f,0,0,0,0,0,0,0};
    CHECK(wubu_ns_novelty(&ns, near, &nov) == 0);
    CHECK(nov < 0.2f);   /* distance to origin ~0.1 */
    CHECK(wubu_ns_evaluate(&ns, near, 5) == 0);   /* below threshold -> rejected */
    CHECK(ns.n_rejected == 2);

    /* a behavior far from origin: novelty large -> archived */
    float far[WUBU_NS_DIM] = {10,0,0,0,0,0,0,0};
    CHECK(wubu_ns_novelty(&ns, far, &nov) == 0);
    CHECK(nov > 9.0f);   /* distance to origin ~10 */
    CHECK(wubu_ns_evaluate(&ns, far, 5) == 1);   /* above threshold -> archived */
    CHECK(ns.n_archived == 2);
    CHECK(wubu_ns_archive_size(&ns) == 2);

    /* now a behavior between origin and far: novelty ~5 (dist to nearer neighbor).
     * {5,0,...}: dist to origin=5, dist to far=5 -> avg=5 >= 5.0 -> archived */
    float mid[WUBU_NS_DIM] = {5,0,0,0,0,0,0,0};
    CHECK(wubu_ns_novelty(&ns, mid, &nov) == 0);
    CHECK(nov >= 4.9f && nov <= 5.1f);
    CHECK(wubu_ns_evaluate(&ns, mid, 5) == 1);
    CHECK(ns.n_archived == 3);

    /* a duplicate of an archived behavior: distance 0 -> novelty 0 -> rejected */
    CHECK(wubu_ns_evaluate(&ns, far, 5) == 0);
    CHECK(ns.n_rejected == 3);

    /* dist helper sanity */
    float a[WUBU_NS_DIM] = {0,0,0,0,0,0,0,0};
    float b[WUBU_NS_DIM] = {3,4,0,0,0,0,0,0};
    CHECK(fabsf(wubu_ns_dist(a,b) - 5.0f) < 1e-5f);

    /* archive capacity: fill then reject */
    wubu_ns_init(&ns, 0.0f);   /* threshold 0 -> everything novel enough */
    float base[WUBU_NS_DIM] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < WUBU_NS_ARCHIVE_MAX; i++) {
        float d[WUBU_NS_DIM]; memcpy(d, base, sizeof(d));
        d[0] = (float)i * 100.0f;   /* spread out -> always novel */
        CHECK(wubu_ns_evaluate(&ns, d, i) == 1);
    }
    CHECK(wubu_ns_archive_size(&ns) == WUBU_NS_ARCHIVE_MAX);
    /* one more -> rejected (full) */
    float d[WUBU_NS_DIM] = {9999,0,0,0,0,0,0,0};
    CHECK(wubu_ns_evaluate(&ns, d, 0) == 0);

    /* bad args */
    CHECK(wubu_ns_novelty(&ns, NULL, &nov) == -1);
    CHECK(wubu_ns_evaluate(NULL, base, 5) == -1);
    CHECK(wubu_ns_seed(&ns, NULL, 5) == -1);
    CHECK(wubu_ns_init(NULL, 1.0f) == -1);
    CHECK(wubu_ns_archive_size(NULL) == -1);
    /* dist with NULL is undefined; ensure no crash only for valid input. */
    float zd[WUBU_NS_DIM] = {0,0,0,0,0,0,0,0};
    CHECK(wubu_ns_dist(zd, zd) == 0.0f);

    if (fails == 0) printf("ALL NOVELTY_SEARCH TESTS PASSED\n");
    else printf("%d NOVELTY_SEARCH TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
