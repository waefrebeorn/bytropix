/* wubu_steam_overlay.c -- Steam Quick Access overlay (Shift+Tab).
 * See wubu_steam_overlay.h. Real, deterministic in-game overlay state machine. */

#include "wubu_steam_overlay.h"
#include <string.h>

int wubu_ov_init(wubu_steam_overlay_t *ov, const char *game_name) {
    if (!ov || !game_name) return -1;
    memset(ov, 0, sizeof(*ov));
    strncpy(ov->game_name, game_name, WUBU_OV_NAME_MAX - 1);
    ov->achievements_total = 0;
    ov->achievements_unlocked = 0;
    ov->focus_index = 0;
    return 0;
}

int wubu_ov_toggle(wubu_steam_overlay_t *ov, unsigned keymask) {
    if (!ov) return -1;
    /* Only Shift+Tab summons/dismisses */
    if ((keymask & (WUBU_OV_KEY_SHIFT | WUBU_OV_KEY_TAB)) !=
        (WUBU_OV_KEY_SHIFT | WUBU_OV_KEY_TAB)) {
        return ov->active;   /* other keys: no change */
    }
    ov->n_toggles++;
    if (!ov->active) {
        /* summon: pause the game, show the main panel */
        ov->active = 1;
        ov->game_paused = 1;
        ov->depth = 1;
        ov->stack[0] = WUBU_OV_PANEL_MAIN;
        ov->focus_index = 0;
    } else {
        /* dismiss: resume the game, clear the panel stack */
        ov->active = 0;
        ov->game_paused = 0;
        ov->depth = 0;
        ov->focus_index = 0;
    }
    return ov->active;
}

int wubu_ov_push(wubu_steam_overlay_t *ov, wubu_ov_panel_t panel) {
    if (!ov || !ov->active) return -1;              /* overlay must be up */
    if (panel <= WUBU_OV_PANEL_MAIN || panel >= WUBU_OV_PANEL_COUNT) return -1;
    if (ov->depth >= WUBU_OV_MAX_STACK) return -1;  /* stack full */
    ov->stack[ov->depth++] = panel;
    ov->focus_index = 0;                            /* focus resets per panel */
    return 0;
}

int wubu_ov_pop(wubu_steam_overlay_t *ov) {
    if (!ov || ov->depth <= 1) return -1;           /* can't pop the root */
    ov->depth--;
    ov->stack[ov->depth] = WUBU_OV_PANEL_NONE;
    ov->focus_index = 0;
    return 0;
}

wubu_ov_panel_t wubu_ov_current(const wubu_steam_overlay_t *ov) {
    if (!ov || ov->depth == 0) return WUBU_OV_PANEL_NONE;
    return ov->stack[ov->depth - 1];
}

int wubu_ov_move_focus(wubu_steam_overlay_t *ov, int delta) {
    if (!ov || !ov->active) return -1;
    int idx = ov->focus_index + delta;
    if (idx < 0) idx = 0;                           /* clamp at top */
    ov->focus_index = idx;
    return 0;
}

int wubu_ov_add_friend(wubu_steam_overlay_t *ov, const char *name,
                       int online, int in_game, const char *playing) {
    if (!ov || !name) return -1;
    if (ov->n_friends >= WUBU_OV_MAX_FRIENDS) return -1;
    /* reject duplicate names */
    for (int i = 0; i < ov->n_friends; i++)
        if (strcmp(ov->friends[i].name, name) == 0) return -1;
    wubu_ov_friend_t *f = &ov->friends[ov->n_friends++];
    memset(f, 0, sizeof(*f));
    strncpy(f->name, name, WUBU_OV_NAME_MAX - 1);
    f->online = online ? 1 : 0;
    f->in_game = in_game ? 1 : 0;
    if (in_game && playing) strncpy(f->playing, playing, WUBU_OV_NAME_MAX - 1);
    f->unread = 0;
    return 0;
}

static int find_friend(wubu_steam_overlay_t *ov, const char *name) {
    for (int i = 0; i < ov->n_friends; i++)
        if (strcmp(ov->friends[i].name, name) == 0) return i;
    return -1;
}

int wubu_ov_friend_message(wubu_steam_overlay_t *ov, const char *name) {
    if (!ov || !name) return -1;
    int i = find_friend(ov, name);
    if (i < 0) return -1;
    ov->friends[i].unread++;
    ov->n_unread_total++;
    return 0;
}

int wubu_ov_unread_total(const wubu_steam_overlay_t *ov) {
    if (!ov) return -1;
    return ov->n_unread_total;
}

int wubu_ov_achievement(wubu_steam_overlay_t *ov) {
    if (!ov) return -1;
    /* can't unlock more than the total */
    if (ov->achievements_total > 0 &&
        ov->achievements_unlocked >= ov->achievements_total) return -1;
    ov->achievements_unlocked++;
    return 0;
}

int wubu_ov_set_achievement_total(wubu_steam_overlay_t *ov, int total) {
    if (!ov || total < 0) return -1;
    ov->achievements_total = total;
    return 0;
}

int wubu_ov_screenshot(wubu_steam_overlay_t *ov) {
    if (!ov) return -1;
    ov->n_screenshots++;
    return 0;
}
