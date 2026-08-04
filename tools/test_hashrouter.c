/*
 * test_hashrouter.c -- standalone tests for wubu_hashrouter.
 *
 * Covers: (1) determinism, (2) top-k distinctness, (3) position
 * sensitivity, (4) load balance (2.5x of uniform, no starvation),
 * (5) output range, (6) arg validation -- plus a second configuration
 * with n_experts=16, top_k=3. Deterministic seed 48.
 *
 * Build & run:
 *   gcc -O2 -std=c11 -Wall -Wextra -I include \
 *       -o /tmp/t_hr tools/test_hashrouter.c src/wubu_hashrouter.c -lm
 *   /tmp/t_hr        # prints ALL PASSED, exit 0
 */
#include "wubu_hashrouter.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_EXPERTS 8
#define TOP_K     2
#define SEED      48u

static int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__); \
        g_failures++; \
    } \
} while (0)

/* exercise one configuration: determinism, range, distinctness,
 * position sensitivity (200-token sweep, >= 90% differ), balance over
 * n_samples (2.5x of uniform, min count > 0). */
static void run_config(int n_experts, int top_k)
{
    int counts[64] = {0};
    int total = 0;
    int diff_pos = 0;
    int det_a[8], det_b[8];

    wubu_hashrouter_t *hr = wubu_hashrouter_create(n_experts, top_k, SEED);
    CHECK(hr != NULL, "create");
    if (!hr) return;

    /* (1) determinism: same (token_id, pos) twice, identical list;
     * a fresh router with the same seed must agree as well */
    CHECK(wubu_hashrouter_route(hr, 123u, 5u, det_a) == top_k, "route returns top_k");
    CHECK(wubu_hashrouter_route(hr, 123u, 5u, det_b) == top_k, "route returns top_k (2nd)");
    CHECK(memcmp(det_a, det_b, (size_t)top_k * sizeof(int)) == 0,
          "determinism across separate calls");
    {
        wubu_hashrouter_t *hr2 = wubu_hashrouter_create(n_experts, top_k, SEED);
        int det_c[8];
        CHECK(hr2 != NULL, "create (fresh)");
        if (hr2) {
            CHECK(wubu_hashrouter_route(hr2, 123u, 5u, det_c) == top_k,
                  "route (fresh)");
            CHECK(memcmp(det_a, det_c, (size_t)top_k * sizeof(int)) == 0,
                  "same seed -> same assignment");
            wubu_hashrouter_free(hr2);
        }
    }

    for (int i = 0; i < 20000; i++) {
        int e[8];
        CHECK(wubu_hashrouter_route(hr, (uint32_t)i, (uint32_t)(i * 7u + 3u), e) == top_k,
              "route in sweep");
        /* (5) range: all outputs in [0, n_experts) */
        for (int k = 0; k < top_k; k++)
            CHECK(e[k] >= 0 && e[k] < n_experts, "range");
        /* (2) distinctness: top_k experts are all different */
        for (int a = 0; a < top_k; a++)
            for (int b = a + 1; b < top_k; b++)
                CHECK(e[a] != e[b], "distinct experts");
        for (int k = 0; k < top_k; k++) counts[e[k]]++;
        total += top_k;

        /* (3) position sensitivity: route(i, i) vs route(i, i+1) */
        if (i < 200) {
            int p0[8], p1[8];
            wubu_hashrouter_route(hr, (uint32_t)i, (uint32_t)i, p0);
            wubu_hashrouter_route(hr, (uint32_t)i, (uint32_t)(i + 1u), p1);
            int same = 1;
            for (int k = 0; k < top_k; k++) {
                int found = 0;
                for (int j = 0; j < top_k; j++)
                    if (p0[k] == p1[j]) { found = 1; break; }
                if (!found) { same = 0; break; }
            }
            if (!same) diff_pos++;
        }
    }
    CHECK(diff_pos >= 180, "position sensitivity >= 90% of 200-token sweep");

    /* (4) balance: every expert within 2.5x of uniform, none starved */
    {
        double uniform = (double)total / (double)n_experts;
        for (int e = 0; e < n_experts; e++) {
            CHECK(counts[e] > 0, "no expert starved");
            CHECK(counts[e] >= 0.4 * uniform && counts[e] <= 2.5 * uniform,
                  "expert share within 2.5x of uniform");
        }
    }

    wubu_hashrouter_free(hr);
}

int main(void)
{
    run_config(N_EXPERTS, TOP_K);        /* 8 experts, top-2 */
    run_config(16, 3);                   /* 16 experts, top-3 */

    /* (6) argument validation */
    CHECK(wubu_hashrouter_create(0, 2, SEED) == NULL, "reject n_experts=0");
    CHECK(wubu_hashrouter_create(8, 0, SEED) == NULL, "reject top_k=0");
    CHECK(wubu_hashrouter_create(8, 9, SEED) == NULL, "reject top_k > n_experts");
    {
        wubu_hashrouter_t *hr = wubu_hashrouter_create(N_EXPERTS, TOP_K, SEED);
        int e[8];
        CHECK(hr != NULL, "create (arg test)");
        CHECK(wubu_hashrouter_route(NULL, 1u, 1u, e) == -1, "route(NULL) -> -1");
        CHECK(wubu_hashrouter_route(hr, 1u, 1u, NULL) == -1, "route(NULL out) -> -1");
        wubu_hashrouter_free(hr);
        wubu_hashrouter_free(NULL);      /* must be a no-op */
    }

    if (g_failures == 0) printf("ALL PASSED\n");
    return g_failures ? 1 : 0;
}
