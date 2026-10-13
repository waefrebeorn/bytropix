/* test_steam_offline.c -- Steam offline mode (cached library play) tests. */
#include "wubu_steam_offline.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_steam_offline_t o;

    CHECK(wubu_off_init(&o) == 0);
    CHECK(o.offline == 0);

    /* add games: two offline-friendly, one always-online */
    CHECK(wubu_off_add_game(&o, "Halo CE", 500, 0) == 0);
    CHECK(wubu_off_add_game(&o, "Portal", 2000, 0) == 1);
    CHECK(wubu_off_add_game(&o, "Siege", 40000, 1) == 2);  /* always-online */
    CHECK(o.n_games == 3);
    CHECK(wubu_off_add_game(&o, "Halo CE", 100, 0) == -1);  /* dup */
    CHECK(wubu_off_add_game(&o, NULL, 1, 0) == -1);
    CHECK(wubu_off_add_game(NULL, "x", 1, 0) == -1);

    /* online mode: installed games launch */
    CHECK(wubu_off_mark_installed(&o, "Halo CE", 1) == 0);
    CHECK(wubu_off_mark_installed(&o, "Portal", 1) == 0);
    CHECK(wubu_off_mark_installed(&o, "Siege", 0) == 0);   /* no cached token */
    CHECK(wubu_off_can_launch(&o, "Halo CE") == 1);
    CHECK(wubu_off_can_launch(&o, "Siege") == 1);          /* online: ok */
    CHECK(wubu_off_can_launch(&o, "nope") == -1);
    CHECK(wubu_off_mark_installed(&o, "nope", 1) == -1);

    /* go offline */
    CHECK(wubu_off_set_mode(&o, 1) == 0);
    CHECK(o.offline == 1);

    /* Halo + Portal: installed + cached + not always-online -> playable */
    CHECK(wubu_off_can_launch(&o, "Halo CE") == 1);
    CHECK(wubu_off_can_launch(&o, "Portal") == 1);
    /* Siege: always-online -> blocked offline */
    CHECK(wubu_off_can_launch(&o, "Siege") == 0);

    /* a game installed but NOT offline-cached -> blocked offline */
    CHECK(wubu_off_add_game(&o, "NoCache", 100, 0) == 3);
    CHECK(wubu_off_mark_installed(&o, "NoCache", 0) == 0);   /* no cached token */
    CHECK(wubu_off_can_launch(&o, "NoCache") == 0);

    /* a not-installed game -> blocked */
    CHECK(wubu_off_add_game(&o, "Uninstalled", 100, 0) == 4);
    CHECK(wubu_off_can_launch(&o, "Uninstalled") == 0);

    /* playable count: Halo + Portal = 2 */
    CHECK(wubu_off_playable_count(&o) == 2);

    /* launch attempts: success + blocked increments */
    CHECK(wubu_off_launch(&o, "Halo CE") == 1);
    CHECK(o.n_blocked == 0);
    CHECK(wubu_off_launch(&o, "Siege") == 0);   /* blocked */
    CHECK(o.n_blocked == 1);
    CHECK(wubu_off_launch(&o, "NoCache") == 0);  /* blocked */
    CHECK(o.n_blocked == 2);
    CHECK(wubu_off_launch(&o, "nope") == -1);

    /* back online: everything installed launches, blocked count unchanged */
    CHECK(wubu_off_set_mode(&o, 0) == 0);
    CHECK(wubu_off_can_launch(&o, "Siege") == 1);
    CHECK(wubu_off_can_launch(&o, "NoCache") == 1);
    CHECK(wubu_off_launch(&o, "Siege") == 1);
    CHECK(o.n_blocked == 2);   /* unchanged */
    CHECK(wubu_off_playable_count(&o) == 4);  /* Halo, Portal, Siege, NoCache */

    /* bad args */
    CHECK(wubu_off_can_launch(NULL, "x") == -1);
    CHECK(wubu_off_can_launch(&o, NULL) == -1);
    CHECK(wubu_off_launch(NULL, "x") == -1);
    CHECK(wubu_off_launch(&o, NULL) == -1);
    CHECK(wubu_off_playable_count(NULL) == -1);
    CHECK(wubu_off_set_mode(NULL, 1) == -1);
    CHECK(wubu_off_init(NULL) == -1);

    if (fails == 0) printf("ALL STEAM_OFFLINE TESTS PASSED\n");
    else printf("%d STEAM_OFFLINE TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
