/* test_broadphase.c -- WORLDSIM #267 broadphase spatial hash grid.
 * Proves: far-apart bodies culled (0 candidates), nearby bodies are
 * candidates, grid clustering reduces pairs vs O(n^2), cell hashing, add/init. */
#include "wubu_broadphase.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_broadphase_t bp;
    CHECK(wubu_bp_init(&bp, 10.0, 0.0, 0.0) == 0, "init cell=10 ok");
    CHECK(wubu_bp_init(&bp, 0.0, 0, 0) == -1, "zero cell rejected");

    /* two bodies far apart (different cells) -> 0 candidates */
    wubu_bp_add(&bp, 5, 5, 1, 0);
    wubu_bp_add(&bp, 95, 95, 1, 1);
    CHECK(wubu_bp_candidate_pairs(&bp) == 0, "far-apart bodies culled (0 candidates)");

    /* two bodies in same cell -> 1 candidate */
    wubu_broadphase_t bp2;
    wubu_bp_init(&bp2, 10.0, 0.0, 0.0);
    wubu_bp_add(&bp2, 5, 5, 1, 0);
    wubu_bp_add(&bp2, 6, 6, 1, 1);
    CHECK(wubu_bp_candidate_pairs(&bp2) == 1, "nearby bodies are candidates");

    /* adjacent cells -> candidate */
    wubu_broadphase_t bp3;
    wubu_bp_init(&bp3, 10.0, 0.0, 0.0);
    wubu_bp_add(&bp3, 9, 5, 1, 0);    /* cell 0 */
    wubu_bp_add(&bp3, 11, 5, 1, 1);   /* cell 1 (adjacent) */
    CHECK(wubu_bp_candidate_pairs(&bp3) == 1, "adjacent-cell bodies are candidates");

    /* clustering: 20 bodies in one cell -> C(20,2)=190 candidates, but a
     * spread-out set culls most. Compare candidate count to O(n^2). */
    wubu_broadphase_t cl;
    wubu_bp_init(&cl, 10.0, 0.0, 0.0);
    for (int i = 0; i < 20; i++) wubu_bp_add(&cl, 1 + (i % 3), 1 + (i % 3), 0.5, i);  /* all in cell 0,0 */
    int cl_pairs = wubu_bp_candidate_pairs(&cl);
    CHECK(cl_pairs == 190, "20 bodies in one cell -> 190 candidates (C(20,2))");

    wubu_broadphase_t sp;
    wubu_bp_init(&sp, 10.0, 0.0, 0.0);
    for (int i = 0; i < 20; i++) wubu_bp_add(&sp, (i % 5) * 20.0, (i / 5) * 20.0, 0.5, i);  /* spread across cells */
    int sp_pairs = wubu_bp_candidate_pairs(&sp);
    CHECK(sp_pairs < cl_pairs, "spread-out set has fewer candidates than clustered");

    /* cell hashing clamps to grid */
    CHECK(wubu_bp_cell_x(&bp, -100) == 0, "negative x clamps to cell 0");
    CHECK(wubu_bp_cell_x(&bp, 100000) == WUBU_BP_GRID - 1, "huge x clamps to last cell");

    /* fail-closed */
    CHECK(wubu_bp_candidate_pairs(NULL) == 0, "NULL -> 0");
    CHECK(wubu_bp_add(NULL, 0, 0, 1, 0) == -1, "NULL add -> -1");

    printf("\n=== test_broadphase: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
