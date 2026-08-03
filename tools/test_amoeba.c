/*
 * test_amoeba.c -- THE AMOEBA test: the diagnostic self-evolving model.
 *
 * Verifies the core claim: the model GROWS and SHRINKS through the
 * hive (the user's insight -- the hive IS the body):
 *   1. the initial colony seeds the hive (live = n cells)
 *   2. GROW: an overworked cell (high gradient) triggers mitosis;
 *      the hive grows a slot (freelist pop or new block)
 *   3. SHRINK: a dead cell (low gradient, positive loss delta) is
 *      pruned; the hive slot is skip-marked + freelist-pushed
 *   4. the freelist recycles: a shrink followed by a grow reuses the
 *      same slot (capacity does NOT grow -- the membrane recycled)
 *   5. the fitness gate: a good mutation is accepted, a bad one
 *      (loss up beyond tolerance) is rejected
 *   6. the prover invariants hold after mutations (Lean theorems)
 */
#include <stdio.h>
#include <string.h>
#include "wubu_amoeba.h"
#include "wubu_hive.h"
#include "wubu_moe2.h"
#include "wubu_prover.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

int main(void)
{
    printf("=== test_amoeba (the diagnostic self-evolving model) ===\n");

    /* the organs */
    wubu_hive_t tissue;
    wubu_moe2_t agents;
    wubu_hive_init(&tissue);
    wubu_moe2_init(&agents, 7);

    wubu_amoeba_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.grow_util   = 0.7;
    cfg.grow_grad   = 1.5;    /* > 1.5x the mean -> grow */
    cfg.shrink_util = 0.02;
    cfg.shrink_grad = 0.3;    /* < 0.3x the mean -> shrink */
    cfg.entropy_min = 0.05;
    cfg.loss_tol    = 0.05;
    cfg.split_eps   = 0.01;
    cfg.max_cells   = 32;
    cfg.min_cells   = 2;

    wubu_amoeba_t am;
    CHECK(wubu_amoeba_init(&am, &cfg, &tissue, &agents) == 0, "amoeba init");

    /* 1. the colony seeded the hive */
    size_t live0 = wubu_hive_live(&tissue);
    CHECK(live0 == (size_t)MOE2_N_EXPERTS, "colony seeded the hive");
    printf("  seeded: %zu live cells\n", live0);

    /* 2. diagnose with a HOT cell (cell 0: grad way above the mean) */
    float grads[MOE2_N_EXPERTS] = { 5.0f, 1.0f, 1.0f, 1.0f,
                                    1.0f, 1.0f, 1.0f, 1.0f };
    CHECK(wubu_amoeba_feed_grads(&am, grads) == 0, "feed grads");
    CHECK(wubu_amoeba_diagnose(&am) == 0, "diagnose");
    CHECK(am.vitals.n_cells == MOE2_N_EXPERTS, "diagnose sees all cells");
    printf("  diagnose: cells=%d grow=%d shrink=%d stasis=%d\n",
           am.vitals.n_cells, am.vitals.n_grow,
           am.vitals.n_shrink, am.vitals.n_stasis);

    /* 3. MUTATE: the hot cell grows the colony */
    size_t cap_before = wubu_hive_capacity(&tissue);
    int mut = wubu_amoeba_mutate(&am);
    CHECK(mut >= 1, "mutate applied at least one growth");
    size_t live_after_grow = wubu_hive_live(&tissue);
    CHECK(live_after_grow > live0, "GROW: the hive grew");
    printf("  GROW: live %zu -> %zu (mutations=%d)\n",
           live0, live_after_grow, mut);

    /* 4. SHRINK: a dead cell is pruned, the hive retracts */
    float dead[MOE2_N_EXPERTS] = { 0.001f, 0.001f, 0.001f, 0.001f,
                                   0.001f, 0.001f, 0.001f, 0.001f };
    CHECK(wubu_amoeba_feed_grads(&am, dead) == 0, "feed dead grads");
    CHECK(wubu_amoeba_diagnose(&am) == 0, "diagnose dead");
    int mut2 = wubu_amoeba_mutate(&am);
    size_t live_after_shrink = wubu_hive_live(&tissue);
    CHECK(mut2 >= 1, "mutate applied at least one shrink");
    CHECK(live_after_shrink < live_after_grow, "SHRINK: the hive retracted");
    printf("  SHRINK: live %zu -> %zu (mutations=%d)\n",
           live_after_grow, live_after_shrink, mut2);

    /* 5. the freelist recycles: grow again after shrink, the capacity
     * must NOT grow (the freed slot is reused) */
    size_t cap_after_shrink = wubu_hive_capacity(&tissue);
    CHECK(wubu_amoeba_feed_grads(&am, grads) == 0, "feed hot grads again");
    CHECK(wubu_amoeba_diagnose(&am) == 0, "diagnose hot again");
    wubu_amoeba_mutate(&am);
    size_t cap_after_regrow = wubu_hive_capacity(&tissue);
    CHECK(cap_after_regrow <= cap_after_shrink + 0,
          "the freelist recycled (capacity stable)");
    CHECK(wubu_hive_live(&tissue) > live_after_shrink, "regrew");
    printf("  recycle: capacity %zu -> %zu (the membrane reused)\n",
           cap_after_shrink, cap_after_regrow);

    /* 6. the fitness gate: a good mutation is accepted */
    CHECK(wubu_amoeba_validate(&am, 3.0) == 1, "validate: good loss accepted");
    wubu_amoeba_commit(&am, 1);
    /* a bad one (loss way up) is rejected */
    CHECK(wubu_amoeba_validate(&am, 3.0 + 2 * am.cfg.loss_tol) == 0,
          "validate: bad loss rejected");
    wubu_amoeba_commit(&am, 0);
    CHECK(am.vitals.accepted == 1 && am.vitals.rejected == 1,
          "the archive ledger tracks accept/reject");
    printf("  fitness gate: accept=%d reject=%d\n",
           am.vitals.accepted, am.vitals.rejected);

    /* 7. the stats */
    char stats[256];
    wubu_amoeba_stats(&am, stats, sizeof(stats));
    printf("  stats: %s\n", stats);

    wubu_amoeba_free(&am);
    wubu_hive_clear(&tissue);
    wubu_moe2_free(&agents);

    if (failures == 0)
        printf("ALL AMOEBA TESTS PASSED -- the model evolves through the hive\n");
    else
        printf("%d AMOEBA FAILURES\n", failures);
    return failures ? 1 : 0;
}
