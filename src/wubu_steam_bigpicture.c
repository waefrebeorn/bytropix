/* wubu_steam_bigpicture.c -- Steam Big Picture / 10-foot library grid.
 * See wubu_steam_bigpicture.h. Real, deterministic gamepad-navigable grid. */

#include "wubu_steam_bigpicture.h"
#include <string.h>

int wubu_bp_init(wubu_steam_bp_t *bp, int cols) {
    if (!bp || cols < 1) return -1;
    memset(bp, 0, sizeof(*bp));
    bp->cols = cols;
    bp->focus = -1;
    bp->view = WUBU_BP_VIEW_GRID;
    return 0;
}

int wubu_bp_add_game(wubu_steam_bp_t *bp, const char *name, const char *hero,
                     int installed, float playtime_hr) {
    if (!bp || !name || !hero) return -1;
    for (int i = 0; i < bp->n_games; i++)
        if (strcmp(bp->games[i].name, name) == 0) return -1;   /* dup */
    if (bp->n_games >= WUBU_BP_MAX_GAMES) return -1;
    wubu_bp_game_t *g = &bp->games[bp->n_games];
    memset(g, 0, sizeof(*g));
    strncpy(g->name, name, WUBU_BP_NAME_MAX - 1);
    strncpy(g->hero, hero, WUBU_BP_NAME_MAX - 1);
    g->installed = installed ? 1 : 0;
    g->playtime_hr = playtime_hr < 0.0f ? 0.0f : playtime_hr;
    int idx = bp->n_games++;
    if (bp->focus < 0) bp->focus = 0;   /* focus first game added */
    return idx;
}

int wubu_bp_move(wubu_steam_bp_t *bp, int drow, int dcol) {
    if (!bp || bp->n_games == 0) return -1;
    int idx = bp->focus;
    if (idx < 0) idx = 0;
    /* grid-aware: down/up move by a full row (cols), left/right by one */
    idx += drow * bp->cols + dcol;
    if (idx < 0) idx = 0;
    if (idx >= bp->n_games) idx = bp->n_games - 1;
    bp->focus = idx;
    return idx;
}

int wubu_bp_open_detail(wubu_steam_bp_t *bp) {
    if (!bp || bp->focus < 0) return -1;
    if (bp->view != WUBU_BP_VIEW_GRID) return -1;   /* already in detail */
    bp->view = WUBU_BP_VIEW_DETAIL;
    return 0;
}

int wubu_bp_back(wubu_steam_bp_t *bp) {
    if (!bp || bp->view != WUBU_BP_VIEW_DETAIL) return -1;
    bp->view = WUBU_BP_VIEW_GRID;
    return 0;
}

int wubu_bp_confirm(wubu_steam_bp_t *bp) {
    if (!bp || bp->focus < 0) return -1;
    if (bp->view == WUBU_BP_VIEW_GRID) {
        bp->view = WUBU_BP_VIEW_DETAIL;
        return 0;   /* detail opened */
    }
    /* in detail: launch the focused game */
    bp->n_launches++;
    return 1;
}

int wubu_bp_focus(const wubu_steam_bp_t *bp) {
    if (!bp) return -1;
    return bp->focus;
}

wubu_bp_view_t wubu_bp_view(const wubu_steam_bp_t *bp) {
    if (!bp) return WUBU_BP_VIEW_GRID;
    return bp->view;
}

const char *wubu_bp_focus_name(const wubu_steam_bp_t *bp) {
    if (!bp || bp->focus < 0 || bp->focus >= bp->n_games) return NULL;
    return bp->games[bp->focus].name;
}
