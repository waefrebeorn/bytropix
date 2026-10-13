/* test_ctr_rt.c -- OMARCHY #291 containerd-style high-level runtime.
 * Proves: full lifecycle (create->start->stop->delete), invalid transitions
 * rejected, cannot delete running task, snapshot accounting, fail-closed. */
#include "wubu_ctr_rt.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_ctr_rt_t rt;
    CHECK(wubu_ctr_init(&rt) == 0, "init ok");

    /* cannot create without a snapshot */
    CHECK(wubu_ctr_create(&rt, "c1") == -1, "create without snapshot -> -1");

    /* pull image (prepares snapshot) */
    CHECK(wubu_ctr_pull_image(&rt) == 0, "pull image");
    CHECK(rt.n_images == 1 && rt.n_snapshots == 1, "1 image, 1 snapshot");

    /* create -> CREATED */
    CHECK(wubu_ctr_create(&rt, "c1") == 0, "create c1");
    CHECK(wubu_ctr_state_of(&rt, "c1") == WUBU_CTR_CREATED, "c1 CREATED");

    /* cannot start twice / cannot stop before start */
    CHECK(wubu_ctr_stop(&rt, "c1") == -1, "stop before start -> -1");
    CHECK(wubu_ctr_start(&rt, "c1") == 0, "start c1");
    CHECK(wubu_ctr_state_of(&rt, "c1") == WUBU_CTR_RUNNING, "c1 RUNNING");
    CHECK(wubu_ctr_start(&rt, "c1") == -1, "double start -> -1");

    /* cannot delete a running task */
    CHECK(wubu_ctr_delete(&rt, "c1") == -1, "delete running -> -1");

    /* stop -> STOPPED */
    CHECK(wubu_ctr_stop(&rt, "c1") == 0, "stop c1");
    CHECK(wubu_ctr_state_of(&rt, "c1") == WUBU_CTR_STOPPED, "c1 STOPPED");
    CHECK(wubu_ctr_stop(&rt, "c1") == -1, "double stop -> -1");

    /* delete -> DELETED, snapshot released */
    CHECK(wubu_ctr_delete(&rt, "c1") == 0, "delete c1");
    CHECK(wubu_ctr_state_of(&rt, "c1") == WUBU_CTR_DELETED, "c1 DELETED");
    CHECK(rt.n_snapshots == 0, "snapshot released on delete");
    CHECK(wubu_ctr_delete(&rt, "c1") == -1, "double delete -> -1");

    /* unknown task */
    CHECK(wubu_ctr_state_of(&rt, "nope") == -1, "unknown task state -> -1");
    CHECK(wubu_ctr_start(&rt, "nope") == -1, "start unknown -> -1");

    /* fail-closed on NULL */
    CHECK(wubu_ctr_init(NULL) == -1, "NULL init -> -1");
    CHECK(wubu_ctr_create(NULL, "x") == -1, "NULL create -> -1");
    CHECK(wubu_ctr_start(NULL, "x") == -1, "NULL start -> -1");
    CHECK(wubu_ctr_delete(NULL, "x") == -1, "NULL delete -> -1");
    CHECK(wubu_ctr_state_of(NULL, "x") == -1, "NULL state -> -1");

    printf("\n=== test_ctr_rt: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
