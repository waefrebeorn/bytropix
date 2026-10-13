/* test_zns.c -- KERNEL #31 tests: ZNS zone state at /n/hw/<dev>/zns. */
#include "wubu_zns.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_zns_t z;

    /* init + add: 4 zones of 100 sectors each */
    CHECK(wubu_zns_init(&z) == 0);
    CHECK(wubu_zns_add(&z, "nvme0", 4, 100) == 0);
    CHECK(wubu_zns_find(&z, "nvme0") == 0);
    CHECK(wubu_zns_add(&z, "nvme0", 4, 100) == -1);       /* dup */
    CHECK(wubu_zns_add(&z, "bad", 0, 100) == -1);         /* n_zones 0 */
    CHECK(wubu_zns_add(&z, "bad", 4, 0) == -1);           /* zone_size 0 */

    /* all zones start EMPTY */
    for (int i = 0; i < 4; i++)
        CHECK(wubu_zns_zone_state(&z, "nvme0", i) == WUBU_ZNS_EMPTY);
    CHECK(wubu_zns_zone_state(&z, "nvme0", 99) == -1);    /* out of range */
    CHECK(wubu_zns_zone_state(&z, "nope", 0) == -1);      /* unknown dev */

    /* state names */
    CHECK(strcmp(wubu_zns_state_name(WUBU_ZNS_EMPTY), "empty") == 0);
    CHECK(strcmp(wubu_zns_state_name(WUBU_ZNS_OPEN), "open") == 0);
    CHECK(strcmp(wubu_zns_state_name(WUBU_ZNS_FULL), "full") == 0);

    /* append requires OPEN */
    CHECK(wubu_zns_append(&z, "nvme0", 0, 10) == -1);     /* empty, not open */

    /* open zone 0, append, watch wp + transition to FULL at capacity */
    CHECK(wubu_zns_open(&z, "nvme0", 0) == 0);
    CHECK(wubu_zns_zone_state(&z, "nvme0", 0) == WUBU_ZNS_OPEN);
    CHECK(wubu_zns_append(&z, "nvme0", 0, 40) == 0);
    CHECK(z.devs[0].zones[0].wp == 40);
    CHECK(wubu_zns_zone_state(&z, "nvme0", 0) == WUBU_ZNS_OPEN);
    CHECK(wubu_zns_append(&z, "nvme0", 0, 60) == 0);      /* wp -> 100 = capacity */
    CHECK(wubu_zns_zone_state(&z, "nvme0", 0) == WUBU_ZNS_FULL);
    CHECK(z.devs[0].open_zone == -1);                     /* auto-closed on full */

    /* overflow fail-closed */
    CHECK(wubu_zns_open(&z, "nvme0", 1) == 0);
    CHECK(wubu_zns_append(&z, "nvme0", 1, 101) == -1);    /* > capacity 100 */
    CHECK(z.devs[0].zones[1].wp == 0);                    /* unchanged */

    /* only one zone open at a time */
    CHECK(wubu_zns_open(&z, "nvme0", 2) == -1);           /* zone 1 still open */

    /* close zone 1 (OPEN -> EMPTY, wp preserved) */
    CHECK(wubu_zns_append(&z, "nvme0", 1, 30) == 0);
    CHECK(wubu_zns_close(&z, "nvme0", 1) == 0);
    CHECK(wubu_zns_zone_state(&z, "nvme0", 1) == WUBU_ZNS_EMPTY);
    CHECK(z.devs[0].zones[1].wp == 30);                   /* data retained */

    /* finish seals a zone FULL */
    CHECK(wubu_zns_open(&z, "nvme0", 2) == 0);
    CHECK(wubu_zns_finish(&z, "nvme0", 2) == 0);
    CHECK(wubu_zns_zone_state(&z, "nvme0", 2) == WUBU_ZNS_FULL);
    CHECK(wubu_zns_finish(&z, "nvme0", 2) == 0);          /* idempotent */

    /* cannot write to a FULL zone */
    CHECK(wubu_zns_append(&z, "nvme0", 2, 1) == -1);
    /* cannot open a FULL zone */
    CHECK(wubu_zns_open(&z, "nvme0", 2) == -1);

    /* reset clears a FULL zone back to EMPTY, wp=0 */
    CHECK(wubu_zns_reset(&z, "nvme0", 2) == 0);
    CHECK(wubu_zns_zone_state(&z, "nvme0", 2) == WUBU_ZNS_EMPTY);
    CHECK(z.devs[0].zones[2].wp == 0);

    /* open is idempotent */
    CHECK(wubu_zns_open(&z, "nvme0", 2) == 0);
    CHECK(wubu_zns_open(&z, "nvme0", 2) == 0);
    CHECK(wubu_zns_close(&z, "nvme0", 2) == 0);

    /* bad-arg fail-closed */
    CHECK(wubu_zns_open(&z, "nvme0", 99) == -1);
    CHECK(wubu_zns_close(&z, "nope", 0) == -1);
    CHECK(wubu_zns_finish(&z, "nvme0", 99) == -1);
    CHECK(wubu_zns_reset(&z, "nope", 0) == -1);
    CHECK(wubu_zns_append(&z, "nvme0", 99, 1) == -1);

    /* second device independent */
    CHECK(wubu_zns_add(&z, "nvme1", 2, 50) == 0);
    CHECK(wubu_zns_open(&z, "nvme1", 0) == 0);
    CHECK(wubu_zns_zone_state(&z, "nvme1", 0) == WUBU_ZNS_OPEN);
    CHECK(wubu_zns_zone_state(&z, "nvme0", 0) == WUBU_ZNS_FULL);  /* nvme0 untouched */

    if (fails == 0) printf("ALL ZNS TESTS PASSED\n");
    else printf("%d ZNS TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
