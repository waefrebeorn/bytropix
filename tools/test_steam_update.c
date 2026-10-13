/* test_steam_update.c -- Steam update channel management tests. */
#include "wubu_steam_update.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_steam_update_t u;

    /* init: stable active, v3.5.9 everywhere */
    CHECK(wubu_up_init(&u) == 0);
    CHECK(u.active == WUBU_UPD_STABLE);
    CHECK(strcmp(wubu_up_version(&u), "3.5.9") == 0);
    CHECK(wubu_up_state(&u) == WUBU_UPD_NONE);
    CHECK(u.auto_update == 1);
    CHECK(strcmp(wubu_up_channel_name(WUBU_UPD_STABLE), "stable") == 0);
    CHECK(strcmp(wubu_up_channel_name(WUBU_UPD_BETA), "beta") == 0);
    CHECK(strcmp(wubu_up_channel_name(WUBU_UPD_PREVIEW), "preview") == 0);
    CHECK(wubu_up_channel_name((wubu_upd_channel_t)99) == NULL);

    /* channel versions can differ (beta ahead of stable) */
    CHECK(wubu_up_set_channel_version(&u, WUBU_UPD_BETA, "3.6.0-beta1") == 0);
    CHECK(wubu_up_set_channel_version(&u, WUBU_UPD_PREVIEW, "3.7.0-preview1") == 0);
    CHECK(wubu_up_set_channel_version(&u, (wubu_upd_channel_t)99, "x") == -1);
    CHECK(wubu_up_set_channel_version(&u, WUBU_UPD_BETA, NULL) == -1);

    /* queue + download + install a version on stable */
    CHECK(wubu_up_queue_version(&u, "3.5.10") == 0);
    CHECK(wubu_up_state(&u) == WUBU_UPD_QUEUED);
    CHECK(wubu_up_download(&u) == 0);
    CHECK(wubu_up_state(&u) == WUBU_UPD_DOWNLOADED);
    CHECK(wubu_up_install(&u) == 0);
    CHECK(wubu_up_state(&u) == WUBU_UPD_INSTALLED);
    CHECK(strcmp(wubu_up_version(&u), "3.5.10") == 0);
    CHECK(u.n_installs == 1);
    CHECK(u.has_snapshot == 1);
    CHECK(strcmp(u.snapshot_version, "3.5.9") == 0);

    /* install with nothing pending: returns 1 */
    CHECK(wubu_up_install(&u) == 1);
    CHECK(wubu_up_download(&u) == 1);

    /* rollback restores the pre-install version */
    CHECK(wubu_up_rollback(&u) == 0);
    CHECK(strcmp(wubu_up_version(&u), "3.5.9") == 0);
    CHECK(u.has_snapshot == 0);
    CHECK(u.n_rollbacks == 1);
    CHECK(wubu_up_state(&u) == WUBU_UPD_NONE);
    /* no snapshot -> rollback fails */
    CHECK(wubu_up_rollback(&u) == -1);

    /* auto-download-before-install: QUEUED -> install auto-downloads */
    CHECK(wubu_up_queue_version(&u, "3.5.11") == 0);
    CHECK(wubu_up_state(&u) == WUBU_UPD_QUEUED);
    CHECK(wubu_up_install(&u) == 0);   /* auto-downloaded then installed */
    CHECK(strcmp(wubu_up_version(&u), "3.5.11") == 0);

    /* with auto_update off, install from QUEUED fails (must download first) */
    CHECK(wubu_up_set_auto(&u, 0) == 0);
    CHECK(wubu_up_queue_version(&u, "3.5.12") == 0);
    CHECK(wubu_up_install(&u) == -1);   /* can't install without download */
    CHECK(wubu_up_download(&u) == 0);
    CHECK(wubu_up_install(&u) == 0);
    CHECK(strcmp(wubu_up_version(&u), "3.5.12") == 0);
    CHECK(wubu_up_set_auto(&u, 1) == 0);

    /* switch channel queues that channel's version */
    CHECK(wubu_up_switch_channel(&u, WUBU_UPD_BETA) == 0);
    CHECK(u.active == WUBU_UPD_BETA);
    CHECK(wubu_up_state(&u) == WUBU_UPD_QUEUED);
    CHECK(strcmp(u.pending_version, "3.6.0-beta1") == 0);
    CHECK(wubu_up_install(&u) == 0);
    CHECK(strcmp(wubu_up_version(&u), "3.6.0-beta1") == 0);

    /* switch to preview */
    CHECK(wubu_up_switch_channel(&u, WUBU_UPD_PREVIEW) == 0);
    CHECK(u.active == WUBU_UPD_PREVIEW);
    CHECK(wubu_up_install(&u) == 0);
    CHECK(strcmp(wubu_up_version(&u), "3.7.0-preview1") == 0);

    /* bad channel */
    CHECK(wubu_up_switch_channel(&u, (wubu_upd_channel_t)99) == -1);
    CHECK(wubu_up_switch_channel(NULL, WUBU_UPD_STABLE) == -1);

    /* bad args */
    CHECK(wubu_up_queue_version(&u, NULL) == -1);
    CHECK(wubu_up_queue_version(NULL, "x") == -1);
    CHECK(wubu_up_download(NULL) == -1);
    CHECK(wubu_up_install(NULL) == -1);
    CHECK(wubu_up_init(NULL) == -1);
    CHECK(wubu_up_set_auto(NULL, 1) == -1);
    CHECK(wubu_up_version(NULL) == NULL);
    CHECK(wubu_up_state(NULL) == WUBU_UPD_NONE);
    CHECK(wubu_up_active_channel(NULL) == WUBU_UPD_STABLE);

    if (fails == 0) printf("ALL STEAM_UPDATE TESTS PASSED\n");
    else printf("%d STEAM_UPDATE TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
