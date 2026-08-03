/* test_bi.c -- Block Importance oracle (ShortGPT, arXiv:2403.03853).
 *
 * DA oracle gates:
 *  1. rank() returns a permutation of 0..L-1
 *  2. shrink_candidate() picks the argmin BI
 *  3. grow_candidate() picks the argmax BI
 *  4. FD robustness: perturbing a low-BI layer's weights by epsilon
 *     must not flip the argmin (the oracle's answer is stable).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_bi.h"

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } \
    else { printf("ok: %s\n", msg); } \
} while (0)

int main(void)
{
    /* synthetic BI array: layer 2 is the most redundant, layer 7 the most critical */
    const int L = 12;
    float bis[12] = {
        0.5f, 0.4f, 0.05f, 0.6f, 0.3f, 0.55f, 0.35f, 0.9f, 0.45f, 0.5f, 0.25f, 0.2f
    };

    /* 1. rank is a permutation of 0..L-1 */
    int *rank = NULL;
    CHECK(wubu_bi_rank(bis, L, &rank) == 0 && rank != NULL, "rank alloc");
    int seen[12] = {0};
    int perm_ok = 1;
    for (int i = 0; i < L; i++) {
        if (rank[i] < 0 || rank[i] >= L || seen[rank[i]]) { perm_ok = 0; break; }
        seen[rank[i]] = 1;
    }
    CHECK(perm_ok, "rank is a permutation");
    CHECK(rank[0] == 2, "rank[0] == layer 2 (lowest BI)");
    CHECK(rank[L-1] == 7, "rank[L-1] == layer 7 (highest BI)");
    free(rank);

    /* 2. shrink candidate = argmin */
    int sc = wubu_bi_shrink_candidate(bis, L, 0.1f);
    CHECK(sc == 2, "shrink_candidate picks layer 2");
    /* threshold above all BIs -> no candidate */
    CHECK(wubu_bi_shrink_candidate(bis, L, -1.0f) == -1, "shrink none if threshold below all");

    /* 3. grow candidate = argmax */
    int gc = wubu_bi_grow_candidate(bis, L, 0.8f);
    CHECK(gc == 7, "grow_candidate picks layer 7");
    CHECK(wubu_bi_grow_candidate(bis, L, 99.0f) == -1, "grow none if threshold above all");

    /* 4. FD robustness: perturb low-BI layer's value, argmin stays */
    float bis_pert[12];
    memcpy(bis_pert, bis, sizeof bis);
    bis_pert[2] = 0.051f; /* +0.001 perturbation */
    CHECK(wubu_bi_shrink_candidate(bis_pert, L, 0.1f) == 2, "argmin stable under +eps");
    bis_pert[2] = 0.049f; /* -0.001 */
    CHECK(wubu_bi_shrink_candidate(bis_pert, L, 0.1f) == 2, "argmin stable under -eps");

    /* 5. edge: single layer has no shrink candidate */
    float single[1] = {1.0f};
    CHECK(wubu_bi_shrink_candidate(single, 1, 0.5f) == -1, "single layer -> no shrink");
    CHECK(wubu_bi_grow_candidate(single, 1, 0.5f) == 0, "single layer -> grow ok");

    if (fails == 0) printf("ALL CLEAN\n");
    else printf("%d FAILURES\n", fails);
    return fails ? 1 : 0;
}
