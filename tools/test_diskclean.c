/* test_diskclean.c -- Win98/XP Disk Cleanup utility tests. */
#include "wubu_diskclean.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_diskclean_t dc;

    CHECK(wubu_dc_init(&dc) == 0);
    CHECK(dc.n_files == 0);
    CHECK(wubu_dc_init(NULL) == -1);

    /* category names */
    CHECK(strcmp(wubu_dc_cat_name(WUBU_DC_TEMP), "Temporary files") == 0);
    CHECK(strcmp(wubu_dc_cat_name(WUBU_DC_CACHE), "Cache") == 0);
    CHECK(strcmp(wubu_dc_cat_name(WUBU_DC_RECYCLE_BIN), "Recycle Bin") == 0);
    CHECK(strcmp(wubu_dc_cat_name(WUBU_DC_LOG), "Logs") == 0);
    CHECK(strcmp(wubu_dc_cat_name(WUBU_DC_DOWNLOAD), "Downloads") == 0);
    CHECK(strcmp(wubu_dc_cat_name(WUBU_DC_OTHER), "Other") == 0);
    CHECK(wubu_dc_cat_name((wubu_dc_cat_t)99) == NULL);

    /* safe categories */
    CHECK(wubu_dc_safe(WUBU_DC_TEMP) == 1);
    CHECK(wubu_dc_safe(WUBU_DC_CACHE) == 1);
    CHECK(wubu_dc_safe(WUBU_DC_RECYCLE_BIN) == 1);
    CHECK(wubu_dc_safe(WUBU_DC_LOG) == 1);
    CHECK(wubu_dc_safe(WUBU_DC_DOWNLOAD) == 0);   /* needs opt-in */
    CHECK(wubu_dc_safe(WUBU_DC_OTHER) == 0);      /* never auto */

    /* default selection = the four safe cats */
    unsigned def = wubu_dc_default_selection();
    CHECK(def & (1u << WUBU_DC_TEMP));
    CHECK(def & (1u << WUBU_DC_CACHE));
    CHECK(def & (1u << WUBU_DC_RECYCLE_BIN));
    CHECK(def & (1u << WUBU_DC_LOG));
    CHECK(!(def & (1u << WUBU_DC_DOWNLOAD)));
    CHECK(!(def & (1u << WUBU_DC_OTHER)));

    /* add files across categories */
    CHECK(wubu_dc_add(&dc, "/tmp/a.tmp", WUBU_DC_TEMP, 1000) == 0);
    CHECK(wubu_dc_add(&dc, "/tmp/b.tmp", WUBU_DC_TEMP, 2000) == 0);
    CHECK(wubu_dc_add(&dc, "/cache/c", WUBU_DC_CACHE, 5000) == 0);
    CHECK(wubu_dc_add(&dc, "/recycle/x", WUBU_DC_RECYCLE_BIN, 3000) == 0);
    CHECK(wubu_dc_add(&dc, "/var/log/app.log", WUBU_DC_LOG, 500) == 0);
    CHECK(wubu_dc_add(&dc, "/home/u/movie.iso", WUBU_DC_DOWNLOAD, 4000000000ULL) == 0);
    CHECK(wubu_dc_add(&dc, "/home/u/notes", WUBU_DC_OTHER, 100) == 0);
    CHECK(dc.n_files == 7);

    /* per-category byte totals + counts */
    CHECK(wubu_dc_cat_bytes(&dc, WUBU_DC_TEMP) == 3000);
    CHECK(wubu_dc_cat_bytes(&dc, WUBU_DC_CACHE) == 5000);
    CHECK(wubu_dc_cat_bytes(&dc, WUBU_DC_RECYCLE_BIN) == 3000);
    CHECK(wubu_dc_cat_bytes(&dc, WUBU_DC_LOG) == 500);
    CHECK(wubu_dc_cat_bytes(&dc, WUBU_DC_DOWNLOAD) == 4000000000ULL);
    CHECK(wubu_dc_cat_bytes(&dc, WUBU_DC_OTHER) == 100);
    CHECK(dc.cat_count[WUBU_DC_TEMP] == 2);
    CHECK(dc.cat_count[WUBU_DC_DOWNLOAD] == 1);
    CHECK(wubu_dc_cat_bytes(&dc, (wubu_dc_cat_t)99) == 0);
    CHECK(wubu_dc_cat_bytes(NULL, WUBU_DC_TEMP) == 0);

    /* reclaimable under default selection (safe cats only) */
    CHECK(wubu_dc_reclaimable(&dc, def) == 3000 + 5000 + 3000 + 500);
    /* default excludes the big download + other */
    CHECK(wubu_dc_reclaimable(&dc, def) < 4000000000ULL);
    /* opt-in download adds its bytes */
    CHECK(wubu_dc_reclaimable(&dc, def | (1u << WUBU_DC_DOWNLOAD)) ==
          3000 + 5000 + 3000 + 500 + 4000000000ULL);
    /* empty selection reclaims nothing */
    CHECK(wubu_dc_reclaimable(&dc, 0) == 0);
    CHECK(wubu_dc_reclaimable(NULL, def) == 0);

    /* bad adds */
    CHECK(wubu_dc_add(&dc, NULL, WUBU_DC_TEMP, 1) == -1);
    CHECK(wubu_dc_add(&dc, "/x", (wubu_dc_cat_t)99, 1) == -1);
    CHECK(wubu_dc_add(NULL, "/x", WUBU_DC_TEMP, 1) == -1);
    /* zero-size file is allowed */
    CHECK(wubu_dc_add(&dc, "/tmp/empty", WUBU_DC_TEMP, 0) == 0);
    CHECK(wubu_dc_cat_bytes(&dc, WUBU_DC_TEMP) == 3000);   /* unchanged */

    if (fails == 0) printf("ALL DISKCLEAN TESTS PASSED\n");
    else printf("%d DISKCLEAN TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
