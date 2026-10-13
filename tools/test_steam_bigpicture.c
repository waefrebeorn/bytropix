/* test_steam_bigpicture.c -- Steam Big Picture / 10-foot library grid tests. */
#include "wubu_steam_bigpicture.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_steam_bp_t bp;

    /* init: 4-column grid */
    CHECK(wubu_bp_init(&bp, 4) == 0);
    CHECK(bp.cols == 4);
    CHECK(bp.focus == -1);
    CHECK(wubu_bp_view(&bp) == WUBU_BP_VIEW_GRID);
    CHECK(wubu_bp_init(&bp, 0) == -1);
    CHECK(wubu_bp_init(&bp, -1) == -1);
    CHECK(wubu_bp_init(NULL, 4) == -1);

    /* add 10 games in a 4-col grid (3 rows) */
    for (int i = 0; i < 10; i++) {
        char name[32]; snprintf(name, sizeof(name), "Game%d", i);
        CHECK(wubu_bp_add_game(&bp, name, "hero", 1, 1.0f * i) == i);
    }
    CHECK(bp.n_games == 10);
    CHECK(wubu_bp_focus(&bp) == 0);   /* first game auto-focused */
    CHECK(strcmp(wubu_bp_focus_name(&bp), "Game0") == 0);
    CHECK(wubu_bp_add_game(&bp, "Game0", "hero", 1, 0) == -1);   /* dup */
    CHECK(wubu_bp_add_game(&bp, NULL, "hero", 1, 0) == -1);
    CHECK(wubu_bp_add_game(&bp, "x", NULL, 1, 0) == -1);
    CHECK(wubu_bp_add_game(NULL, "x", "h", 1, 0) == -1);

    /* move right: 0 -> 1 -> 2 -> 3 */
    CHECK(wubu_bp_move(&bp, 0, 1) == 1);
    CHECK(wubu_bp_move(&bp, 0, 1) == 2);
    CHECK(wubu_bp_move(&bp, 0, 1) == 3);
    /* move down: row +1 -> 3 + 4 = 7 */
    CHECK(wubu_bp_move(&bp, 1, 0) == 7);
    /* move right: 7 -> 8 */
    CHECK(wubu_bp_move(&bp, 0, 1) == 8);
    /* move down clamps at the last game (8 + 4 = 12 -> clamp 9) */
    CHECK(wubu_bp_move(&bp, 1, 0) == 9);
    /* move up: 9 - 4 = 5 */
    CHECK(wubu_bp_move(&bp, -1, 0) == 5);
    /* move left clamps at 0 */
    CHECK(wubu_bp_move(&bp, 0, -100) == 0);

    /* detail open/back */
    CHECK(wubu_bp_open_detail(&bp) == 0);
    CHECK(wubu_bp_view(&bp) == WUBU_BP_VIEW_DETAIL);
    CHECK(wubu_bp_open_detail(&bp) == -1);   /* already in detail */
    CHECK(wubu_bp_back(&bp) == 0);
    CHECK(wubu_bp_view(&bp) == WUBU_BP_VIEW_GRID);
    CHECK(wubu_bp_back(&bp) == -1);   /* already in grid */

    /* confirm in grid opens detail */
    CHECK(wubu_bp_confirm(&bp) == 0);
    CHECK(wubu_bp_view(&bp) == WUBU_BP_VIEW_DETAIL);
    /* confirm in detail launches */
    CHECK(wubu_bp_confirm(&bp) == 1);
    CHECK(bp.n_launches == 1);
    CHECK(wubu_bp_confirm(&bp) == 1);   /* launch again from detail */
    CHECK(bp.n_launches == 2);
    /* back to grid, confirm opens detail again */
    CHECK(wubu_bp_back(&bp) == 0);
    CHECK(wubu_bp_confirm(&bp) == 0);
    CHECK(wubu_bp_view(&bp) == WUBU_BP_VIEW_DETAIL);

    /* focus name in detail */
    CHECK(strcmp(wubu_bp_focus_name(&bp), "Game0") == 0);

    /* move while in detail still changes focus (grid nav works) */
    CHECK(wubu_bp_move(&bp, 0, 1) == 1);
    CHECK(strcmp(wubu_bp_focus_name(&bp), "Game1") == 0);

    /* empty library: move/confirm fail */
    wubu_steam_bp_t empty;
    CHECK(wubu_bp_init(&empty, 4) == 0);
    CHECK(wubu_bp_move(&empty, 0, 1) == -1);
    CHECK(wubu_bp_confirm(&empty) == -1);
    CHECK(wubu_bp_open_detail(&empty) == -1);
    CHECK(wubu_bp_focus_name(&empty) == NULL);
    CHECK(wubu_bp_focus(&empty) == -1);

    /* bad args */
    CHECK(wubu_bp_move(NULL, 0, 1) == -1);
    CHECK(wubu_bp_confirm(NULL) == -1);
    CHECK(wubu_bp_open_detail(NULL) == -1);
    CHECK(wubu_bp_back(NULL) == -1);
    CHECK(wubu_bp_focus(NULL) == -1);
    CHECK(wubu_bp_view(NULL) == WUBU_BP_VIEW_GRID);
    CHECK(wubu_bp_focus_name(NULL) == NULL);

    if (fails == 0) printf("ALL STEAM_BIGPICTURE TESTS PASSED\n");
    else printf("%d STEAM_BIGPICTURE TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
