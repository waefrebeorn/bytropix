/* test_nvmeqd.c -- KERNEL #27 tests: NVMe queue depth tunable via /n. */
#include "wubu_nvmeqd.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_nvmeqd_t q;

    /* init */
    CHECK(wubu_qd_init(&q) == 0);
    CHECK(q.n == 0);

    /* add a device with bounds [2,64], initial depth 32 */
    CHECK(wubu_qd_add(&q, "nvme0", 2, 64, 32) == 0);
    CHECK(wubu_qd_find(&q, "nvme0") == 0);
    CHECK(wubu_qd_get(&q, "nvme0") == 32);

    /* duplicate name rejected */
    CHECK(wubu_qd_add(&q, "nvme0", 2, 64, 8) == -1);

    /* bad bounds rejected */
    CHECK(wubu_qd_add(&q, "bad1", 0, 64, 8) == -1);   /* min < 1 */
    CHECK(wubu_qd_add(&q, "bad2", 64, 2, 8) == -1);   /* max < min */
    CHECK(wubu_qd_add(&q, "bad3", 2, 64, 100) == -1); /* init out of range */

    /* get on unknown device */
    CHECK(wubu_qd_get(&q, "nope") == -1);
    CHECK(wubu_qd_find(&q, "nope") == -1);

    /* set within bounds (the /n write path) */
    CHECK(wubu_qd_set(&q, "nvme0", 16) == 0);
    CHECK(wubu_qd_get(&q, "nvme0") == 16);
    CHECK(wubu_qd_set(&q, "nvme0", 2) == 0);   /* min edge */
    CHECK(wubu_qd_set(&q, "nvme0", 64) == 0);  /* max edge */

    /* set out of range -> fail-closed, depth unchanged */
    CHECK(wubu_qd_set(&q, "nvme0", 1) == -1);   /* below min */
    CHECK(wubu_qd_get(&q, "nvme0") == 64);
    CHECK(wubu_qd_set(&q, "nvme0", 65) == -1);  /* above max */
    CHECK(wubu_qd_get(&q, "nvme0") == 64);

    /* set on unknown device */
    CHECK(wubu_qd_set(&q, "nope", 8) == -1);

    /* version bumps only on change */
    unsigned long long v0 = wubu_qd_version(&q, "nvme0");
    CHECK(wubu_qd_set(&q, "nvme0", 32) == 0);   /* 64 -> 32, a change */
    CHECK(wubu_qd_version(&q, "nvme0") == v0 + 1);
    CHECK(wubu_qd_set(&q, "nvme0", 32) == 0);   /* 32 -> 32, no change */
    CHECK(wubu_qd_version(&q, "nvme0") == v0 + 1);

    /* in-flight accounting clamped to [0, io_depth] */
    CHECK(wubu_qd_inflight(&q, "nvme0", 0) == 0);
    CHECK(wubu_qd_inflight(&q, "nvme0", 10) == 10);
    CHECK(wubu_qd_inflight(&q, "nvme0", 100) == 32);  /* clamped to depth */
    CHECK(wubu_qd_inflight(&q, "nvme0", -50) == 0);   /* clamped to 0 */

    /* set fails while commands are in flight (non-idle) */
    CHECK(wubu_qd_inflight(&q, "nvme0", 4) == 4);
    CHECK(wubu_qd_set(&q, "nvme0", 8) == -1);   /* non-idle -> fail-closed */
    CHECK(wubu_qd_get(&q, "nvme0") == 32);
    /* drain, then set succeeds */
    CHECK(wubu_qd_inflight(&q, "nvme0", -4) == 0);
    CHECK(wubu_qd_set(&q, "nvme0", 8) == 0);
    CHECK(wubu_qd_get(&q, "nvme0") == 8);

    /* coherence check */
    CHECK(wubu_qd_coherent(&q.devs[0]) == 1);
    q.devs[0].sq_id = 2;  /* break SQ/CQ pairing */
    CHECK(wubu_qd_coherent(&q.devs[0]) == 0);
    q.devs[0].sq_id = 1;

    /* registry full */
    for (int i = 1; i < WUBU_QD_MAX_DEVS; i++) {
        char nm[16]; snprintf(nm, sizeof nm, "nvme%d", i);
        CHECK(wubu_qd_add(&q, nm, 1, 128, 64) == 0);
    }
    CHECK(wubu_qd_add(&q, "overflow", 1, 128, 64) == -1);

    if (fails == 0) printf("ALL NVMeQD TESTS PASSED\n");
    else printf("%d NVMeQD TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
