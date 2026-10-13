/* test_steam_overlay.c -- Steam Quick Access overlay (Shift+Tab) tests. */
#include "wubu_steam_overlay.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)
#define SHTAB (WUBU_OV_KEY_SHIFT | WUBU_OV_KEY_TAB)

int main(void)
{
    wubu_steam_overlay_t ov;

    /* init for a running game: hidden, not paused */
    CHECK(wubu_ov_init(&ov, "Halo CE") == 0);
    CHECK(ov.active == 0);
    CHECK(ov.game_paused == 0);
    CHECK(wubu_ov_current(&ov) == WUBU_OV_PANEL_NONE);

    /* non-Shift+Tab keys do nothing */
    CHECK(wubu_ov_toggle(&ov, WUBU_OV_KEY_SHIFT) == 0);
    CHECK(wubu_ov_toggle(&ov, WUBU_OV_KEY_TAB) == 0);
    CHECK(ov.active == 0);

    /* Shift+Tab summons: game pauses, main panel shown */
    CHECK(wubu_ov_toggle(&ov, SHTAB) == 1);
    CHECK(ov.active == 1);
    CHECK(ov.game_paused == 1);
    CHECK(ov.depth == 1);
    CHECK(wubu_ov_current(&ov) == WUBU_OV_PANEL_MAIN);
    CHECK(ov.n_toggles == 1);

    /* push a sub-panel */
    CHECK(wubu_ov_push(&ov, WUBU_OV_PANEL_FRIENDS) == 0);
    CHECK(ov.depth == 2);
    CHECK(wubu_ov_current(&ov) == WUBU_OV_PANEL_FRIENDS);
    /* can't push the root or an invalid panel */
    CHECK(wubu_ov_push(&ov, WUBU_OV_PANEL_MAIN) == -1);
    CHECK(wubu_ov_push(&ov, WUBU_OV_PANEL_NONE) == -1);
    CHECK(wubu_ov_push(&ov, (wubu_ov_panel_t)999) == -1);

    /* pop back to main */
    CHECK(wubu_ov_pop(&ov) == 0);
    CHECK(wubu_ov_current(&ov) == WUBU_OV_PANEL_MAIN);
    /* can't pop the root */
    CHECK(wubu_ov_pop(&ov) == -1);

    /* push fails while hidden */
    CHECK(wubu_ov_toggle(&ov, SHTAB) == 0);   /* dismiss */
    CHECK(ov.active == 0);
    CHECK(ov.game_paused == 0);
    CHECK(wubu_ov_current(&ov) == WUBU_OV_PANEL_NONE);
    CHECK(wubu_ov_push(&ov, WUBU_OV_PANEL_FRIENDS) == -1);
    CHECK(wubu_ov_pop(&ov) == -1);
    CHECK(wubu_ov_move_focus(&ov, 1) == -1);

    /* focus navigation while up */
    CHECK(wubu_ov_toggle(&ov, SHTAB) == 1);
    CHECK(ov.focus_index == 0);
    CHECK(wubu_ov_move_focus(&ov, 1) == 0);
    CHECK(wubu_ov_move_focus(&ov, 1) == 0);
    CHECK(ov.focus_index == 2);
    CHECK(wubu_ov_move_focus(&ov, -5) == 0);   /* clamps to 0 */
    CHECK(ov.focus_index == 0);
    /* focus resets when entering a sub-panel */
    ov.focus_index = 3;
    CHECK(wubu_ov_push(&ov, WUBU_OV_PANEL_ACHIEVEMENTS) == 0);
    CHECK(ov.focus_index == 0);

    /* friends */
    CHECK(wubu_ov_add_friend(&ov, "alice", 1, 1, "Portal 2") == 0);
    CHECK(wubu_ov_add_friend(&ov, "bob", 1, 0, "") == 0);
    CHECK(wubu_ov_add_friend(&ov, "carol", 0, 0, "") == 0);
    CHECK(ov.n_friends == 3);
    CHECK(wubu_ov_add_friend(&ov, "alice", 1, 0, "") == -1);  /* dup */
    CHECK(ov.friends[0].online == 1 && ov.friends[0].in_game == 1);
    CHECK(strcmp(ov.friends[0].playing, "Portal 2") == 0);
    CHECK(wubu_ov_add_friend(NULL, "x", 1, 0, "") == -1);
    CHECK(wubu_ov_add_friend(&ov, NULL, 1, 0, "") == -1);

    /* messages increment unread */
    CHECK(wubu_ov_unread_total(&ov) == 0);
    CHECK(wubu_ov_friend_message(&ov, "alice") == 0);
    CHECK(wubu_ov_friend_message(&ov, "alice") == 0);
    CHECK(wubu_ov_friend_message(&ov, "bob") == 0);
    CHECK(ov.friends[0].unread == 2);
    CHECK(ov.n_unread_total == 3);
    CHECK(wubu_ov_friend_message(&ov, "nobody") == -1);
    CHECK(wubu_ov_friend_message(NULL, "alice") == -1);
    CHECK(wubu_ov_unread_total(NULL) == -1);

    /* achievements: unlock bounded by total */
    CHECK(wubu_ov_set_achievement_total(&ov, 3) == 0);
    CHECK(ov.achievements_total == 3);
    CHECK(wubu_ov_achievement(&ov) == 0);
    CHECK(wubu_ov_achievement(&ov) == 0);
    CHECK(ov.achievements_unlocked == 2);
    CHECK(wubu_ov_achievement(&ov) == 0);   /* 3rd ok, reaches total */
    CHECK(wubu_ov_achievement(&ov) == -1);  /* 4th rejected */
    CHECK(wubu_ov_set_achievement_total(&ov, -1) == -1);
    CHECK(wubu_ov_achievement(NULL) == -1);

    /* no total set -> unbounded unlock */
    wubu_steam_overlay_t ov2;
    CHECK(wubu_ov_init(&ov2, "Game") == 0);
    CHECK(wubu_ov_achievement(&ov2) == 0);
    CHECK(wubu_ov_achievement(&ov2) == 0);
    CHECK(ov2.achievements_unlocked == 2);

    /* screenshots */
    CHECK(ov.n_screenshots == 0);
    CHECK(wubu_ov_screenshot(&ov) == 0);
    CHECK(wubu_ov_screenshot(&ov) == 0);
    CHECK(ov.n_screenshots == 2);
    CHECK(wubu_ov_screenshot(NULL) == -1);

    /* toggle count and resume-on-dismiss */
    CHECK(wubu_ov_toggle(&ov, SHTAB) == 0);   /* dismiss again */
    CHECK(ov.game_paused == 0);
    CHECK(wubu_ov_current(&ov) == WUBU_OV_PANEL_NONE);
    CHECK(ov.n_toggles == 4);   /* summon, dismiss, re-summon, dismiss */

    if (fails == 0) printf("ALL STEAM_OVERLAY TESTS PASSED\n");
    else printf("%d STEAM_OVERLAY TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
