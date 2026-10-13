/* test_steam_workshop.c -- Steam Workshop / mod manager tests. */
#include "wubu_steam_workshop.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_steam_workshop_t w;

    CHECK(wubu_ws_init(&w) == 0);
    CHECK(w.n_mods == 0);
    CHECK(wubu_ws_find(&w, 1) == -1);

    /* subscribe two mods */
    uint64_t m1 = wubu_ws_subscribe(&w, "HD Textures", "alice");
    uint64_t m2 = wubu_ws_subscribe(&w, "Better AI", "bob");
    CHECK(m1 == 1 && m2 == 2);
    CHECK(w.n_mods == 2);
    CHECK(wubu_ws_find(&w, m1) == 0);
    CHECK(wubu_ws_dl_state(&w, m1) == WUBU_WS_SUBSCRIBED);
    CHECK(wubu_ws_subscribe(&w, NULL, "x") == (uint64_t)-1);
    CHECK(wubu_ws_subscribe(NULL, "x", "y") == (uint64_t)-1);

    /* load requires downloaded + enabled */
    CHECK(wubu_ws_load(&w, m1) == -1);   /* not downloaded */
    CHECK(wubu_ws_download_start(&w, m1) == 0);
    CHECK(wubu_ws_dl_state(&w, m1) == WUBU_WS_DOWNLOADING);
    CHECK(wubu_ws_download_done(&w, m1) == 0);
    CHECK(wubu_ws_dl_state(&w, m1) == WUBU_WS_DOWNLOADED);
    CHECK(wubu_ws_download_start(&w, m1) == -1);   /* already downloaded */
    CHECK(wubu_ws_load(&w, m1) == 0);
    CHECK(wubu_ws_n_loaded(&w) == 1);
    /* double-load is a no-op (0) */
    CHECK(wubu_ws_load(&w, m1) == 0);
    CHECK(wubu_ws_n_loaded(&w) == 1);

    /* disable then load fails */
    uint64_t m3 = wubu_ws_subscribe(&w, "Mod3", "carol");
    CHECK(wubu_ws_download_start(&w, m3) == 0);
    CHECK(wubu_ws_download_done(&w, m3) == 0);
    CHECK(wubu_ws_set_enabled(&w, m3, 0) == 0);
    CHECK(wubu_ws_load(&w, m3) == -1);   /* disabled */
    CHECK(wubu_ws_set_enabled(&w, m3, 1) == 0);
    CHECK(wubu_ws_load(&w, m3) == 0);
    CHECK(wubu_ws_n_loaded(&w) == 2);

    /* download fail path */
    uint64_t m4 = wubu_ws_subscribe(&w, "Mod4", "dave");
    CHECK(wubu_ws_download_start(&w, m4) == 0);
    CHECK(wubu_ws_download_fail(&w, m4) == 0);
    CHECK(wubu_ws_dl_state(&w, m4) == WUBU_WS_FAILED);
    CHECK(wubu_ws_load(&w, m4) == -1);
    /* retry from failed */
    CHECK(wubu_ws_download_start(&w, m4) == 0);
    CHECK(wubu_ws_download_done(&w, m4) == 0);
    CHECK(wubu_ws_load(&w, m4) == 0);
    CHECK(wubu_ws_n_loaded(&w) == 3);

    /* reorder: move m1 (idx0) by +2 swaps with idx2 -> [m4, m3, m1] */
    CHECK(wubu_ws_reorder(&w, m1, 2) == 0);
    CHECK(w.load_order[0] == m4);
    CHECK(w.load_order[1] == m3);
    CHECK(w.load_order[2] == m1);
    /* move m1 (idx2) by -2 swaps with idx0 -> [m1, m3, m4] */
    CHECK(wubu_ws_reorder(&w, m1, -2) == 0);
    CHECK(w.load_order[0] == m1);
    CHECK(w.load_order[2] == m4);
    /* reorder out of range fails */
    CHECK(wubu_ws_reorder(&w, m1, -1) == -1);   /* can't go before 0 */
    CHECK(wubu_ws_reorder(&w, m1, 99) == -1);
    CHECK(wubu_ws_reorder(&w, 999, 1) == -1);   /* not loaded */

    /* dependencies */
    CHECK(wubu_ws_add_dep(&w, m2, m1) == 0);    /* m2 requires m1 */
    CHECK(wubu_ws_add_dep(&w, m2, m1) == 0);    /* dup dep ok */
    CHECK(wubu_ws_add_dep(&w, m2, 999) == -1);  /* dep must exist */
    CHECK(wubu_ws_add_dep(&w, m2, m2) == -1);   /* no self-dep */
    CHECK(wubu_ws_add_dep(&w, 999, m1) == -1);  /* bad mod */
    CHECK(wubu_ws_add_dep(NULL, m2, m1) == -1);
    CHECK(w.mods[wubu_ws_find(&w, m2)].n_deps == 1);

    /* conflict detection: two loaded mods providing the same file */
    CHECK(wubu_ws_add_file(&w, m1, "textures/rock.dds") == 0);
    CHECK(wubu_ws_add_file(&w, m3, "textures/rock.dds") == 0);  /* conflict w/ m1 */
    CHECK(wubu_ws_add_file(&w, m4, "textures/sand.dds") == 0);  /* unique */
    CHECK(wubu_ws_count_conflicts(&w) == 1);   /* m1 vs m3 share rock.dds */

    /* disabling m3 removes its conflict */
    CHECK(wubu_ws_set_enabled(&w, m3, 0) == 0);
    CHECK(wubu_ws_count_conflicts(&w) == 0);
    CHECK(wubu_ws_set_enabled(&w, m3, 1) == 0);
    CHECK(wubu_ws_count_conflicts(&w) == 1);

    /* unloading m3 removes its conflict */
    CHECK(wubu_ws_unload(&w, m3) == 0);
    CHECK(wubu_ws_n_loaded(&w) == 2);
    CHECK(wubu_ws_count_conflicts(&w) == 0);
    CHECK(wubu_ws_unload(&w, m3) == -1);   /* already unloaded */

    /* add_file bad args + capacity */
    CHECK(wubu_ws_add_file(&w, 999, "x") == -1);
    CHECK(wubu_ws_add_file(&w, m1, NULL) == -1);
    CHECK(wubu_ws_add_file(NULL, m1, "x") == -1);

    /* unsubscribe removes mod + load-order entry */
    CHECK(wubu_ws_unsubscribe(&w, m1) == 0);
    CHECK(wubu_ws_find(&w, m1) == -1);
    CHECK(wubu_ws_find(&w, m3) >= 0);   /* m3 still there */
    CHECK(wubu_ws_n_loaded(&w) == 1);   /* only m4 remains loaded */
    CHECK(wubu_ws_unsubscribe(&w, 999) == -1);
    CHECK(wubu_ws_unsubscribe(NULL, m1) == -1);

    /* misc bad args */
    CHECK(wubu_ws_find(NULL, 1) == -1);
    CHECK(wubu_ws_dl_state(NULL, 1) == WUBU_WS_NONE);
    CHECK(wubu_ws_n_loaded(NULL) == -1);
    CHECK(wubu_ws_count_conflicts(NULL) == -1);
    CHECK(wubu_ws_load(NULL, 1) == -1);
    CHECK(wubu_ws_download_start(NULL, 1) == -1);
    CHECK(wubu_ws_init(NULL) == -1);

    if (fails == 0) printf("ALL STEAM_WORKSHOP TESTS PASSED\n");
    else printf("%d STEAM_WORKSHOP TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
