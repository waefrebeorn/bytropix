#ifndef WUBU_GAMEPAD_NAV_H
#define WUBU_GAMEPAD_NAV_H

/* wubu_gamepad_nav.h -- Steam gamepad virtual cursor + radial menu.
 *
 * The gamepad navigation layer for the 10-foot (Big Picture) UI: a virtual
 * cursor driven by the left analog stick, snap-to-focus targets, and a radial
 * (pie) menu summoned by a trigger/hotkey that supports directional selection
 * (up/down/left/right) + confirm/cancel. Includes a virtual on-screen keyboard
 * for text entry (D-pad grid navigation + character confirm). Deterministic
 * state machine, no stubs. */

#include <stddef.h>

#define WUBU_GN_MAX_TARGETS 32
#define WUBU_GN_LABEL_MAX 32
#define WUBU_GN_SCREEN_W 1280
#define WUBU_GN_SCREEN_H 800
#define WUBU_GN_KB_ROWS 6
#define WUBU_GN_KB_COLS 10

/* Radial menu sectors (triggered navigation) */
typedef enum {
    WUBU_GN_SECTOR_NONE = 0,
    WUBU_GN_SECTOR_UP,
    WUBU_GN_SECTOR_DOWN,
    WUBU_GN_SECTOR_LEFT,
    WUBU_GN_SECTOR_RIGHT,
    WUBU_GN_SECTOR_COUNT
} wubu_gn_sector_t;

typedef struct {
    int x, y;                       /* center */
    char label[WUBU_GN_LABEL_MAX];
    int enabled;
} wubu_gn_target_t;

typedef struct {
    /* virtual cursor */
    int cx, cy;                     /* cursor position (screen coords) */
    int focus;                      /* focused target index, -1 if none */
    wubu_gn_target_t targets[WUBU_GN_MAX_TARGETS];
    int n_targets;
    int speed;                      /* cursor speed px per stick tick */
    int cursor_visible;

    /* radial menu */
    int radial_open;
    wubu_gn_sector_t radial_last;   /* last selected sector */
    int n_radial_selects;

    /* virtual keyboard */
    int kb_open;
    int kb_row, kb_col;             /* highlighted key */
    char kb_text[256];              /* composed text */
    int kb_len;
} wubu_gamepad_nav_t;

/* Initialise (cursor centered, no focus, speed 8). */
int wubu_gn_init(wubu_gamepad_nav_t *gn);

/* Add a focus target (button/menu item). Returns index or -1. */
int wubu_gn_add_target(wubu_gamepad_nav_t *gn, int x, int y, const char *label);

/* Move the virtual cursor by a stick delta; re-snaps focus to the nearest
 * enabled target within a snap radius. Returns the new focus index or -1. */
int wubu_gn_move_cursor(wubu_gamepad_nav_t *gn, int dx, int dy);

/* Set the focus directly (D-pad snap). Returns 0/-1. */
int wubu_gn_set_focus(wubu_gamepad_nav_t *gn, int idx);

/* Confirm the focused target (A button). Returns 0/-1 (nothing focused). */
int wubu_gn_confirm(wubu_gamepad_nav_t *gn);

/* Radial menu: open/close. */
int wubu_gn_radial_open(wubu_gamepad_nav_t *gn);
int wubu_gn_radial_close(wubu_gamepad_nav_t *gn);

/* Select a radial sector. Returns 0/-1 (menu closed / bad sector). */
int wubu_gn_radial_select(wubu_gamepad_nav_t *gn, wubu_gn_sector_t sector);

/* Virtual keyboard */
int wubu_gn_kb_open(wubu_gamepad_nav_t *gn);
int wubu_gn_kb_close(wubu_gamepad_nav_t *gn);
int wubu_gn_kb_move(wubu_gamepad_nav_t *gn, int drow, int dcol);
int wubu_gn_kb_press(wubu_gamepad_nav_t *gn);   /* confirm highlighted key */

/* The keyboard grid character for a (row,col). Returns a char or 0 if empty.
 * Row 5 is special: col 0 = space, col 1 = backspace, col 9 = done. */
char wubu_gn_kb_char(int row, int col);

/* Accessors */
int wubu_gn_focus(const wubu_gamepad_nav_t *gn);
const char *wubu_gn_focus_label(const wubu_gamepad_nav_t *gn);
const char *wubu_gn_kb_text(const wubu_gamepad_nav_t *gn);

#endif
