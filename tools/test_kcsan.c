/* test_kcsan.c -- KERNEL #29 tests: KCSAN data-race detector. */
#include "wubu_kcsan.h"
#include <stdio.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_kcsan_t k;
    wubu_kcsan_race_t r;

    CHECK(wubu_kcsan_init(&k) == 0);
    CHECK(wubu_kcsan_race_count(&k) == 0);

    /* same-thread accesses never race */
    CHECK(wubu_kcsan_access(&k, 0x1000, 8, 1, 1) == 0);   /* T1 write */
    CHECK(wubu_kcsan_access(&k, 0x1000, 8, 1, 1) == 0);   /* T1 write again */
    CHECK(wubu_kcsan_race_count(&k) == 0);

    /* write-write race across threads */
    CHECK(wubu_kcsan_access(&k, 0x2000, 8, 1, 1) == 0);   /* T1 write */
    CHECK(wubu_kcsan_access(&k, 0x2000, 8, 1, 2) == 1);   /* T2 write -> RACE */
    CHECK(wubu_kcsan_race_count(&k) == 1);
    CHECK(wubu_kcsan_get_race(&k, 0, &r) == 0);
    CHECK(r.addr == 0x2000 && r.is_write == 1);
    CHECK((r.thread_a == 1 && r.thread_b == 2) || (r.thread_a == 2 && r.thread_b == 1));

    /* read-write race */
    CHECK(wubu_kcsan_access(&k, 0x3000, 8, 0, 1) == 0);   /* T1 read */
    CHECK(wubu_kcsan_access(&k, 0x3000, 8, 1, 2) == 1);   /* T2 write -> RACE */
    CHECK(wubu_kcsan_race_count(&k) == 2);

    /* read-read does NOT race */
    CHECK(wubu_kcsan_access(&k, 0x4000, 8, 0, 1) == 0);
    CHECK(wubu_kcsan_access(&k, 0x4000, 8, 0, 2) == 0);   /* both reads */
    CHECK(wubu_kcsan_race_count(&k) == 2);                /* unchanged */

    /* non-overlapping addresses never race */
    CHECK(wubu_kcsan_access(&k, 0x5000, 8, 1, 1) == 0);
    CHECK(wubu_kcsan_access(&k, 0x5008, 8, 1, 2) == 0);   /* disjoint */
    CHECK(wubu_kcsan_race_count(&k) == 2);

    /* overlapping (partial) ranges race */
    CHECK(wubu_kcsan_access(&k, 0x6000, 16, 1, 1) == 0);
    CHECK(wubu_kcsan_access(&k, 0x6008, 16, 1, 2) == 1);  /* overlap [6008,6010) */
    CHECK(wubu_kcsan_race_count(&k) == 3);

    /* barrier clears watches: accesses separated by a barrier never race */
    CHECK(wubu_kcsan_barrier(&k) == 0);
    CHECK(wubu_kcsan_access(&k, 0x7000, 8, 1, 1) == 0);   /* T1 write */
    CHECK(wubu_kcsan_barrier(&k) == 0);                   /* happens-before edge */
    CHECK(wubu_kcsan_access(&k, 0x7000, 8, 1, 2) == 0);   /* T2 write, post-barrier */
    CHECK(wubu_kcsan_race_count(&k) == 3);                /* unchanged */

    /* size-0 access is a no-op */
    CHECK(wubu_kcsan_access(&k, 0x8000, 0, 1, 1) == 0);

    /* NULL fail-closed */
    CHECK(wubu_kcsan_init(NULL) == -1);
    CHECK(wubu_kcsan_access(NULL, 0x1000, 8, 1, 1) == -1);
    CHECK(wubu_kcsan_race_count(NULL) == 0);
    CHECK(wubu_kcsan_get_race(&k, 999, &r) == -1);
    CHECK(wubu_kcsan_get_race(&k, -1, &r) == -1);

    /* a fresh write after a consumed race re-arms the watchpoint */
    CHECK(wubu_kcsan_access(&k, 0x9000, 8, 1, 1) == 0);
    CHECK(wubu_kcsan_access(&k, 0x9000, 8, 1, 2) == 1);   /* race #4 */
    CHECK(wubu_kcsan_race_count(&k) == 4);
    CHECK(wubu_kcsan_access(&k, 0x9000, 8, 1, 3) == 1);   /* T3 vs the re-armed T2 watch */
    CHECK(wubu_kcsan_race_count(&k) == 5);

    if (fails == 0) printf("ALL KCSAN TESTS PASSED\n");
    else printf("%d KCSAN TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
