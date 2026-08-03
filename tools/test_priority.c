/* test_priority.c -- the priority store (AGI institutional memory).
 *
 * DA oracle gates:
 *  1. init clears everything
 *  2. should_shrink: BI above critical bar -> never shrink
 *  3. should_shrink: BI above redundant threshold -> never shrink
 *  4. should_grow: rolled-back grow -> never grow again (shame list)
 *  5. should_shrink: rolled-back shrink -> never shrink again
 *  6. log_event updates mutation/rollback counters
 *  7. save/load round-trip preserves the ledger
 *  8. load of garbage is rejected (initialized guard)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_priority.h"

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } \
    else { printf("ok: %s\n", msg); } \
} while (0)

int main(void)
{
    wubu_priority_t p;
    wubu_priority_init(&p);

    /* 1. init */
    CHECK(p.initialized == 1 && p.n_events == 0 && p.n_layers == 0,
          "init clears the store");

    /* 2-3. BI gating */
    float bis[8] = {0.1f, 0.2f, 0.9f, 0.3f, 0.05f, 0.4f, 0.6f, 0.15f};
    CHECK(wubu_priority_set_bi(&p, bis, 8) == 0, "set_bi ok");
    CHECK(p.n_layers == 8, "set_bi records n_layers");
    /* layer 2 has BI 0.9 -> critical, never shrink */
    CHECK(wubu_priority_should_shrink(&p, 2, 0.1f, 0.8f) == 0,
          "critical layer (BI 0.9 >= bar 0.8) never shrinks");
    /* layer 0 has BI 0.1 <= threshold 0.1 and < bar -> shrink ok */
    CHECK(wubu_priority_should_shrink(&p, 0, 0.1f, 0.8f) == 1,
          "redundant layer (BI 0.1 <= 0.1) shrinks");
    /* layer 1 has BI 0.2 > threshold 0.1 -> not redundant enough */
    CHECK(wubu_priority_should_shrink(&p, 1, 0.1f, 0.8f) == 0,
          "mid layer (BI 0.2 > 0.1) does not shrink");

    /* 4-5. shame list */
    CHECK(wubu_priority_should_grow(&p, 3) == 1, "grow layer 3 allowed (no history)");
    wubu_priority_log_event(&p, WUBU_PRI_EVT_GROW, 3, 1.0f, 1.5f, 0); /* rolled back */
    CHECK(wubu_priority_should_grow(&p, 3) == 0, "grow layer 3 blocked (was rolled back)");
    CHECK(wubu_priority_should_grow(&p, 4) == 1, "grow layer 4 still allowed");

    wubu_priority_log_event(&p, WUBU_PRI_EVT_SHRINK, 0, 1.0f, 0.7f, 1); /* accepted */
    CHECK(wubu_priority_should_shrink(&p, 0, 0.1f, 0.8f) == 1,
          "shrink layer 0 still allowed (accepted shrink is not shame)");

    /* 6. counters */
    CHECK(p.mutation_count == 1, "mutation_count == 1 (one accepted grow/shrink)");
    CHECK(p.rollback_count == 1, "rollback_count == 1");
    CHECK(p.n_events == 2, "two events logged");

    /* 7. save/load round-trip */
    p.step = 424242;
    CHECK(wubu_priority_save(&p, "/tmp/wubu_priority_test.bin") == 0, "save ok");
    wubu_priority_t q;
    wubu_priority_init(&q);
    CHECK(wubu_priority_load(&q, "/tmp/wubu_priority_test.bin") == 0, "load ok");
    CHECK(q.n_events == 2 && q.mutation_count == 1 && q.rollback_count == 1,
          "round-trip preserves ledger");
    CHECK(q.step == 424242, "round-trip preserves step");
    CHECK(wubu_priority_should_grow(&q, 3) == 0, "round-trip preserves shame list");

    /* 8. garbage rejected */
    FILE *f = fopen("/tmp/wubu_priority_garbage.bin", "wb");
    fwrite("NOT A PRIORITY STORE..............", 1, 40, f);
    fclose(f);
    wubu_priority_t r;
    wubu_priority_init(&r);
    CHECK(wubu_priority_load(&r, "/tmp/wubu_priority_garbage.bin") == -1,
          "garbage load rejected");
    CHECK(r.initialized == 1 && r.n_events == 0, "store reset after bad load");

    remove("/tmp/wubu_priority_test.bin");
    remove("/tmp/wubu_priority_garbage.bin");

    if (fails == 0) printf("ALL CLEAN\n");
    else printf("%d FAILURES\n", fails);
    return fails ? 1 : 0;
}
