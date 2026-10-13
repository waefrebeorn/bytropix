/* test_steam_cloud.c -- Steam Cloud Saves sync tests. */
#include "wubu_steam_cloud.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_steam_cloud_t c;

    /* init: online, 1 GiB, newer-wins */
    CHECK(wubu_cl_init(&c) == 0);
    CHECK(c.online == 1);
    CHECK(c.policy == WUBU_CLOUD_RESOLVE_NEWER);
    CHECK(c.quota_bytes == (1024u*1024u*1024u));

    /* add games */
    CHECK(wubu_cl_add_game(&c, "Halo", 5000) == 0);
    CHECK(wubu_cl_add_game(&c, "Portal", 3000) == 1);
    CHECK(c.n_games == 2);
    CHECK(c.used_bytes == 8000);
    CHECK(wubu_cl_add_game(&c, "Halo", 100) == -1);   /* dup */
    CHECK(wubu_cl_add_game(&c, NULL, 100) == -1);
    CHECK(wubu_cl_add_game(NULL, "x", 1) == -1);

    /* fresh game: no local, no cloud */
    CHECK(wubu_cl_state(&c, "Halo") == WUBU_CLOUD_MISSING_LOCAL);
    CHECK(wubu_cl_state(&c, "nope") == WUBU_CLOUD_MISSING_LOCAL);

    /* local-only save -> UPLOAD state */
    CHECK(wubu_cl_save_local(&c, "Halo", 1000) == 0);
    CHECK(wubu_cl_state(&c, "Halo") == WUBU_CLOUD_UPLOAD);
    /* sync pushes to cloud */
    CHECK(wubu_cl_sync(&c, "Halo") == 0);
    CHECK(c.games[0].has_cloud == 1);
    CHECK(c.n_uploads == 1);
    CHECK(wubu_cl_state(&c, "Halo") == WUBU_CLOUD_UP_TO_DATE);
    /* re-sync is a no-op */
    CHECK(wubu_cl_sync(&c, "Halo") == 1);

    /* cloud-only save -> DOWNLOAD */
    CHECK(wubu_cl_save_cloud(&c, "Portal", 2000) == 0);
    CHECK(wubu_cl_state(&c, "Portal") == WUBU_CLOUD_DOWNLOAD);
    CHECK(wubu_cl_sync(&c, "Portal") == 0);
    CHECK(c.games[1].has_local == 1);
    CHECK(c.n_downloads == 1);
    CHECK(wubu_cl_state(&c, "Portal") == WUBU_CLOUD_UP_TO_DATE);

    /* local changed since base -> UPLOAD */
    CHECK(wubu_cl_save_local(&c, "Halo", 3000) == 0);
    CHECK(wubu_cl_state(&c, "Halo") == WUBU_CLOUD_UPLOAD);
    CHECK(wubu_cl_sync(&c, "Halo") == 0);
    CHECK(c.n_uploads == 2);

    /* cloud changed since base -> DOWNLOAD */
    CHECK(wubu_cl_save_cloud(&c, "Halo", 4000) == 0);
    CHECK(wubu_cl_state(&c, "Halo") == WUBU_CLOUD_DOWNLOAD);
    CHECK(wubu_cl_sync(&c, "Halo") == 0);
    CHECK(c.n_downloads == 2);

    /* CONFLICT: both changed since last sync. base=2, local=3, cloud=3 */
    CHECK(wubu_cl_save_local(&c, "Halo", 5000) == 0);   /* local v3 */
    CHECK(wubu_cl_save_cloud(&c, "Halo", 6000) == 0);   /* cloud v3 */
    CHECK(wubu_cl_state(&c, "Halo") == WUBU_CLOUD_CONFLICT);
    CHECK(c.n_conflicts == 0);
    /* newer-wins: cloud ts 6000 > local ts 5000 -> download wins */
    CHECK(wubu_cl_sync(&c, "Halo") == 0);
    CHECK(c.n_conflicts == 1);
    CHECK(c.n_resolved == 1);
    CHECK(c.n_downloads == 3);   /* cloud won */
    CHECK(wubu_cl_state(&c, "Halo") == WUBU_CLOUD_UP_TO_DATE);

    /* CONFLICT keep-local policy */
    c.policy = WUBU_CLOUD_RESOLVE_KEEP_LOCAL;
    CHECK(wubu_cl_save_local(&c, "Halo", 7000) == 0);
    CHECK(wubu_cl_save_cloud(&c, "Halo", 8000) == 0);
    CHECK(wubu_cl_state(&c, "Halo") == WUBU_CLOUD_CONFLICT);
    int dl_before = c.n_downloads, ul_before = c.n_uploads;
    CHECK(wubu_cl_sync(&c, "Halo") == 0);
    CHECK(c.n_uploads == ul_before + 1);   /* local kept -> pushed to cloud */
    CHECK(c.n_downloads == dl_before);

    /* CONFLICT keep-cloud policy */
    c.policy = WUBU_CLOUD_RESOLVE_KEEP_CLOUD;
    CHECK(wubu_cl_save_local(&c, "Halo", 9000) == 0);
    CHECK(wubu_cl_save_cloud(&c, "Halo", 10000) == 0);
    dl_before = c.n_downloads; ul_before = c.n_uploads;
    CHECK(wubu_cl_sync(&c, "Halo") == 0);
    CHECK(c.n_downloads == dl_before + 1);   /* cloud kept -> pulled */
    CHECK(c.n_uploads == ul_before);

    /* offline: sync fails */
    wubu_cl_set_online(&c, 0);
    CHECK(wubu_cl_save_local(&c, "Halo", 11000) == 0);
    CHECK(wubu_cl_sync(&c, "Halo") == -1);
    CHECK(wubu_cl_sync_all(&c) == -1);
    wubu_cl_set_online(&c, 1);

    /* sync_all: only auto-sync games */
    CHECK(wubu_cl_set_auto(&c, "Portal", 0) == 0);   /* disable Portal */
    CHECK(wubu_cl_save_local(&c, "Halo", 12000) == 0);   /* needs upload */
    CHECK(wubu_cl_save_local(&c, "Portal", 12000) == 0); /* disabled, skipped */
    int synced = wubu_cl_sync_all(&c);
    CHECK(synced == 1);   /* only Halo */
    CHECK(wubu_cl_state(&c, "Halo") == WUBU_CLOUD_UP_TO_DATE);
    CHECK(wubu_cl_state(&c, "Portal") != WUBU_CLOUD_UP_TO_DATE);  /* still pending */
    CHECK(wubu_cl_set_auto(&c, "nope", 1) == -1);

    /* quota */
    CHECK(wubu_cl_quota_exceeded(&c, 1000) == 0);
    CHECK(wubu_cl_quota_exceeded(&c, c.quota_bytes) == 1);   /* used+quota > quota */
    CHECK(wubu_cl_quota_exceeded(NULL, 1) == -1);

    /* bad args */
    CHECK(wubu_cl_save_local(&c, NULL, 1) == -1);
    CHECK(wubu_cl_save_local(&c, "nope", 1) == -1);
    CHECK(wubu_cl_save_cloud(&c, "nope", 1) == -1);
    CHECK(wubu_cl_sync(&c, "nope") == -1);
    CHECK(wubu_cl_sync(NULL, "Halo") == -1);
    CHECK(wubu_cl_init(NULL) == -1);
    wubu_cl_set_online(NULL, 1);   /* no crash */

    if (fails == 0) printf("ALL STEAM_CLOUD TESTS PASSED\n");
    else printf("%d STEAM_CLOUD TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
