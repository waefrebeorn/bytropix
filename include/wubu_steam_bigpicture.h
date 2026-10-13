#ifndef WUBU_STEAM_BIGPICTURE_H
#define WUBU_STEAM_BIGPICTURE_H

/* wubu_steam_bigpicture.h -- Steam Big Picture / 10-foot library grid.
 *
 * Models the Big Picture UI: a gamepad-navigable grid of game tiles over a
 * library. The selection moves with D-pad directions (grid-aware: up/down
 * move by column count, left/right by one), a focused tile opens a detail
 * panel, and launching the focused game is the confirm action. Deterministic,
 * no stubs. */

#include <stddef.h>

#define WUBU_BP_MAX_GAMES 128
#define WUBU_BP_NAME_MAX 48

typedef enum {
    WUBU_BP_VIEW_GRID = 0,     /* browsing the tile grid */
    WUBU_BP_VIEW_DETAIL        /* focused game's detail panel open */
} wubu_bp_view_t;

typedef struct {
    char name[WUBU_BP_NAME_MAX];
    char hero[WUBU_BP_NAME_MAX];   /* hero-art description / capsule text */
    int installed;
    float playtime_hr;
} wubu_bp_game_t;

typedef struct {
    wubu_bp_game_t games[WUBU_BP_MAX_GAMES];
    int n_games;
    int cols;                 /* grid columns */
    int focus;                /* focused game index, -1 if none */
    wubu_bp_view_t view;
    int n_launches;           /* games launched from Big Picture */
} wubu_steam_bp_t;

/* Initialise with a grid column count (>=1). */
int wubu_bp_init(wubu_steam_bp_t *bp, int cols);

/* Add a game tile. Returns index or -1 (dup/full). */
int wubu_bp_add_game(wubu_steam_bp_t *bp, const char *name, const char *hero,
                     int installed, float playtime_hr);

/* Move the grid selection. `drow`/`dcol` are grid deltas (up/down move by
 * `cols`). Returns the new focus index, -1 if no games. */
int wubu_bp_move(wubu_steam_bp_t *bp, int drow, int dcol);

/* Open the focused game's detail panel. Returns 0/-1. */
int wubu_bp_open_detail(wubu_steam_bp_t *bp);

/* Back to the grid from the detail panel. Returns 0/-1. */
int wubu_bp_back(wubu_steam_bp_t *bp);

/* Confirm (A button): in grid, opens detail; in detail, launches the game.
 * Returns 1 if a game launched, 0 if detail opened, -1 on bad args/none. */
int wubu_bp_confirm(wubu_steam_bp_t *bp);

/* Query. */
int wubu_bp_focus(const wubu_steam_bp_t *bp);
wubu_bp_view_t wubu_bp_view(const wubu_steam_bp_t *bp);
const char *wubu_bp_focus_name(const wubu_steam_bp_t *bp);

#endif
