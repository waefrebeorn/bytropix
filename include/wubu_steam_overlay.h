#ifndef WUBU_STEAM_OVERLAY_H
#define WUBU_STEAM_OVERLAY_H

/* wubu_steam_overlay.h -- Steam Quick Access overlay (Shift+Tab).
 *
 * The signature SteamOS/Steam Deck UX feature: an in-game overlay summoned
 * with Shift+Tab while a game is running. The overlay suspends the game
 * (pause), shows a navigable panel stack (main menu -> sub-panels), tracks
 * notification badges (friend messages, achievements), and dismisses on
 * Shift+Tab again (resume). Gamepad-navigable (D-pad/AB), but keyboard
 * equivalents work too. Deterministic state machine, no stubs. */

#include <stddef.h>

/* Overlay panels (the Shift+Tab main menu destinations) */
typedef enum {
    WUBU_OV_PANEL_NONE = 0,
    WUBU_OV_PANEL_MAIN,        /* the main Quick Access menu */
    WUBU_OV_PANEL_FRIENDS,     /* friends list + chat */
    WUBU_OV_PANEL_ACHIEVEMENTS,/* achievement progress */
    WUBU_OV_PANEL_GUIDES,      /* community guides */
    WUBU_OV_PANEL_BROWSER,     /* in-game web browser */
    WUBU_OV_PANEL_SETTINGS,    /* overlay settings */
    WUBU_OV_PANEL_SCREENSHOTS, /* screenshot manager */
    WUBU_OV_PANEL_COUNT
} wubu_ov_panel_t;

#define WUBU_OV_MAX_STACK 8
#define WUBU_OV_MAX_FRIENDS 32
#define WUBU_OV_NAME_MAX 48
#define WUBU_OV_KEY_SHIFT (1u<<0)
#define WUBU_OV_KEY_TAB   (1u<<1)

typedef struct {
    char name[WUBU_OV_NAME_MAX];
    int online;              /* 1 online, 0 offline */
    int in_game;             /* in a game */
    char playing[WUBU_OV_NAME_MAX]; /* game name if in_game */
    int unread;              /* unread message count */
} wubu_ov_friend_t;

typedef struct {
    int active;              /* overlay visible */
    int game_paused;         /* game suspended while overlay up */
    wubu_ov_panel_t stack[WUBU_OV_MAX_STACK];
    int depth;               /* panel stack depth (0 = hidden) */
    wubu_ov_friend_t friends[WUBU_OV_MAX_FRIENDS];
    int n_friends;
    int n_unread_total;      /* sum of unread across friends */
    int n_screenshots;       /* screenshots taken this session */
    int achievements_unlocked;
    int achievements_total;
    char game_name[WUBU_OV_NAME_MAX];
    int n_toggles;           /* Shift+Tab invocations */
    int focus_index;         /* focused menu item in current panel */
} wubu_steam_overlay_t;

/* Initialise for a running game (overlay hidden, game not paused). */
int wubu_ov_init(wubu_steam_overlay_t *ov, const char *game_name);

/* The Shift+Tab hotkey: summon or dismiss the overlay. Returns the new
 * active state (1 shown, 0 hidden), -1 on bad args. */
int wubu_ov_toggle(wubu_steam_overlay_t *ov, unsigned keymask);

/* Navigate into a sub-panel (push). Returns 0/-1. */
int wubu_ov_push(wubu_steam_overlay_t *ov, wubu_ov_panel_t panel);

/* Back to the previous panel (pop). Returns 0/-1 (fails at root). */
int wubu_ov_pop(wubu_steam_overlay_t *ov);

/* Current panel (top of stack), or PANEL_NONE if hidden. */
wubu_ov_panel_t wubu_ov_current(const wubu_steam_overlay_t *ov);

/* Move focus within the current panel (delta can be negative). */
int wubu_ov_move_focus(wubu_steam_overlay_t *ov, int delta);

/* Friends */
int wubu_ov_add_friend(wubu_steam_overlay_t *ov, const char *name,
                       int online, int in_game, const char *playing);
int wubu_ov_friend_message(wubu_steam_overlay_t *ov, const char *name);
int wubu_ov_unread_total(const wubu_steam_overlay_t *ov);

/* Achievements + screenshots (report progress through the overlay) */
int wubu_ov_achievement(wubu_steam_overlay_t *ov);          /* unlock one */
int wubu_ov_set_achievement_total(wubu_steam_overlay_t *ov, int total);
int wubu_ov_screenshot(wubu_steam_overlay_t *ov);           /* F12 */

#endif
