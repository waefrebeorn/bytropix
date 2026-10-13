/* test_snapshot.c -- OMARCHY #273/#274 btrfs subvolume snapshot + rollback.
 * Proves: create, first is active, rollback changes active, find by id,
 * retention pruning, timeline cadence, auto-naming, fail-closed on NULL. */
#include "wubu_snapshot.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_snapmgr_t m;
    CHECK(wubu_snap_init(&m, "@root", 0) == 0, "init ok (unlimited retention)");
    CHECK(wubu_snap_count(&m) == 0, "empty count 0");
    CHECK(wubu_snap_active(&m) == NULL, "no active yet");

    /* create first snapshot -> becomes active */
    uint64_t s1 = wubu_snap_create(&m, "boot", WUBU_SNAP_MANUAL, 1000);
    CHECK(s1 > 0, "create s1");
    CHECK(wubu_snap_count(&m) == 1, "count 1");
    const wubu_snapshot_t *a = wubu_snap_active(&m);
    CHECK(a != NULL && a->id == s1, "first snapshot active");
    CHECK(strcmp(a->name, "boot") == 0, "named 'boot'");

    /* create more */
    uint64_t s2 = wubu_snap_create(&m, "pre-update", WUBU_SNAP_PRE, 2000);
    uint64_t s3 = wubu_snap_create(&m, NULL, WUBU_SNAP_POST, 3000);   /* auto-named */
    CHECK(wubu_snap_count(&m) == 3, "count 3");
    CHECK(wubu_snap_active(&m)->id == s1, "still s1 active (create doesn't change active)");

    /* auto-naming for s3 */
    const wubu_snapshot_t *s3s = wubu_snap_find(&m, s3);
    CHECK(s3s != NULL && s3s->name[0] != '\0', "s3 auto-named non-empty");
    CHECK(s3s->type == WUBU_SNAP_POST, "s3 type POST");

    /* find by id */
    CHECK(wubu_snap_find(&m, s2) != NULL, "find s2");
    CHECK(wubu_snap_find(&m, 99999) == NULL, "find unknown -> NULL");

    /* rollback to s2 */
    CHECK(wubu_snap_rollback(&m, s2) == 0, "rollback to s2");
    CHECK(wubu_snap_active(&m)->id == s2, "s2 now active");
    CHECK(wubu_snap_rollback(&m, 99999) == -1, "rollback unknown -> -1");

    /* rollback to s3, then back to s1 */
    wubu_snap_rollback(&m, s3);
    CHECK(wubu_snap_active(&m)->id == s3, "s3 active");
    wubu_snap_rollback(&m, s1);
    CHECK(wubu_snap_active(&m)->id == s1, "s1 active again");

    /* retention pruning: keep 3. Active is r1 (oldest) which must be
     * preserved; create r4, r5. The active snapshot is never dropped, so the
     * count can reach retention+1 while the oldest (active) is held. */
    wubu_snapmgr_t r;
    wubu_snap_init(&r, "@root", 3);
    uint64_t r1 = wubu_snap_create(&r, "a", WUBU_SNAP_MANUAL, 100);
    uint64_t r2 = wubu_snap_create(&r, "b", WUBU_SNAP_MANUAL, 200);
    uint64_t r3 = wubu_snap_create(&r, "c", WUBU_SNAP_MANUAL, 300);
    CHECK(wubu_snap_count(&r) == 3, "retention: 3 kept");
    uint64_t r4 = wubu_snap_create(&r, "d", WUBU_SNAP_MANUAL, 400);
    CHECK(wubu_snap_count(&r) <= 4, "retention caps near 3 (active preserved)");
    CHECK(wubu_snap_find(&r, r1) != NULL, "active (oldest) snapshot preserved");
    CHECK(wubu_snap_find(&r, r4) != NULL, "newest retained");

    /* after rolling the active target forward, the oldest is prunable */
    wubu_snap_rollback(&r, r4);
    uint64_t r5 = wubu_snap_create(&r, "e", WUBU_SNAP_MANUAL, 500);
    CHECK(wubu_snap_count(&r) <= 3, "retention enforces 3 once oldest is not active");
    CHECK(wubu_snap_find(&r, r5) != NULL, "newest retained after prune");
    (void)r2; (void)r3;

    /* timeline cadence */
    wubu_snapmgr_t t;
    wubu_snap_init(&t, "@root", 0);
    CHECK(wubu_snap_timeline(&t, 1000, 100) == 1, "first timeline created");
    CHECK(wubu_snap_timeline(&t, 1050, 100) == 0, "not due yet (1050 < 1100)");
    CHECK(wubu_snap_timeline(&t, 1100, 100) == 1, "due at 1100");
    CHECK(wubu_snap_timeline(&t, 1150, 100) == 0, "not due yet");
    const wubu_snapshot_t *tl = wubu_snap_find(&t, t.next_id - 1);
    CHECK(tl != NULL && tl->type == WUBU_SNAP_TIMELINE, "timeline snapshot type");

    /* fail-closed on NULL */
    CHECK(wubu_snap_create(NULL, "x", WUBU_SNAP_MANUAL, 1) == 0, "NULL create -> 0");
    CHECK(wubu_snap_rollback(NULL, 1) == -1, "NULL rollback -> -1");
    CHECK(wubu_snap_active(NULL) == NULL, "NULL active -> NULL");
    CHECK(wubu_snap_count(NULL) == 0, "NULL count -> 0");

    printf("\n=== test_snapshot: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
