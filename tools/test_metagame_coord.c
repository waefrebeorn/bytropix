/*
 * test_metagame_coord.c -- AH01-AH04 (coord) + AH05/AH06/AH08/AH13 (metagame)
 *                         + AH12 (credit) verification.
 */
#include "wubu_coord.h"
#include "wubu_metagame.h"
#include "wubu_credit.h"
#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_metagame_coord (AH01-04,05,06,08,12,13) ===\n");

    /* AH01 intent-lock */
    wubu_locktab_t lt; memset(&lt, 0, sizeof lt);
    CHECK(wubu_lock_acquire(&lt, "src/wubu_model.c", "cog", 1000, 30000) == 1, "cog locks model.c");
    CHECK(wubu_lock_acquire(&lt, "src/wubu_model.c", "other", 1001, 30000) == 0, "other DENIED (held by cog)");
    CHECK(wubu_lock_acquire(&lt, "src/wubu_model.c", "cog", 1002, 30000) == 1, "cog re-lock ok");
    CHECK(wubu_lock_release(&lt, "src/wubu_model.c", "cog") == 1, "cog releases");
    CHECK(wubu_lock_acquire(&lt, "src/wubu_model.c", "other", 1003, 30000) == 1, "other now locks (freed)");
    /* stale adoption */
    wubu_locktab_t lt2; memset(&lt2, 0, sizeof lt2);
    wubu_lock_acquire(&lt2, "src/x.c", "ghost", 0, 30000);
    CHECK(wubu_lock_acquire(&lt2, "src/x.c", "cog", 40000, 30000) == 1, "stale lock (40s>30s) adopted by cog");

    /* AH02 transaction commit */
    wubu_locktab_t lt3; memset(&lt3, 0, sizeof lt3);
    wubu_lock_acquire(&lt3, "a.c", "cog", 1, 30000);
    wubu_lock_acquire(&lt3, "b.c", "cog", 1, 30000);
    const char *fs[2] = { "a.c", "b.c" };
    CHECK(wubu_txn_committable(&lt3, fs, 2, "cog") == 1, "both files owned -> committable");
    wubu_lock_release(&lt3, "b.c", "cog");
    CHECK(wubu_txn_committable(&lt3, fs, 2, "cog") == 0, "b.c lost -> NOT committable (repair)");

    /* AH03 shared-memory ACL (default deny) */
    wubu_memacl_t acl; memset(&acl, 0, sizeof acl);
    strncpy(acl.agents[0], "cog", 31); acl.n = 1;
    CHECK(wubu_mem_allowed(&acl, "cog") == 1, "cog authorized");
    CHECK(wubu_mem_allowed(&acl, "other") == 0, "other denied (default deny)");

    /* AH04 conflict resolution + heartbeat */
    CHECK(strcmp(wubu_resolve_conflict("cog", 1, "other", 5), "cog") == 0, "higher priority (cog=1) wins");
    CHECK(strcmp(wubu_resolve_conflict("cog", 9, "other", 2), "other") == 0, "lower priority yields");
    CHECK(wubu_heartbeat_alive(1000, 1020, 30000) == 1, "alive within window");
    CHECK(wubu_heartbeat_alive(1000, 40000, 30000) == 0, "stale -> not alive");

    /* AH05 archive (keeps weak variants for diversity) */
    wubu_archive_t ar; memset(&ar, 0, sizeof ar);
    wubu_archive_add(&ar, "v1", "seed", 20.0, 1);
    wubu_archive_add(&ar, "v2", "v1", 12.0, 1);  /* weak but kept */
    wubu_archive_add(&ar, "v3", "v1", 50.0, 1);
    CHECK(ar.n == 3, "three variants archived (weak kept)");
    CHECK(wubu_archive_best(&ar) == 50.0, "best fitness = 50 (v3)");

    /* AH06 + AH08 + AH13 acceptance */
    CHECK(wubu_accept_child(&ar, "v4", 55.0, 1) == 1, "verified + fit>0 -> accept");
    CHECK(wubu_accept_child(&ar, "v5", 55.0, 0) == 0, "UNVERIFIED self-log -> REJECT (AH08)");
    CHECK(wubu_improvement_delta(50.0, 20.0, 1.0) == 1, "positive delta v3>v1");
    CHECK(wubu_improvement_delta(21.0, 20.0, 5.0) == 0, "delta < min_gain -> reject");

    /* AH12 turn-level credit (TRACE-style TD) */
    double c1 = wubu_turn_credit(0.2, 0.7, 0.0, 0.9); /* predictability rose */
    CHECK(c1 > 0, "positive credit when progress rose");
    double c2 = wubu_turn_credit(0.7, 0.2, 0.0, 0.9); /* derailed */
    CHECK(c2 < 0, "negative credit when progress fell");
    double cr[2] = { c1, c2 };
    CHECK(wubu_credit_sign(c1, 1e-3) == 1 && wubu_credit_sign(c2, 1e-3) == -1, "sign classified");
    CHECK(wubu_credit_sum(cr, 2) < 0, "sum reflects net (one good one bad)");

    if (failures == 0) { printf("ALL METAGAME-COORD TESTS PASSED\n"); return 0; }
    printf("%d METAGAME-COORD TEST(S) FAILED\n", failures);
    return 1;
}
