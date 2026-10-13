/* wubu_gamepad_nav.c -- Steam gamepad virtual cursor + radial menu.
 * See wubu_gamepad_nav.h. Real, deterministic gamepad navigation layer. */

#include "wubu_gamepad_nav.h"
#include <string.h>

/* The on-screen keyboard grid (rows 0..4 = letters/digits, row 5 = controls) */
static const char g_kb[WUBU_GN_KB_ROWS][WUBU_GN_KB_COLS] = {
    {'1','2','3','4','5','6','7','8','9','0'},
    {'q','w','e','r','t','y','u','i','o','p'},
    {'a','s','d','f','g','h','j','k','l','-'},
    {'z','x','c','v','b','n','m','.',',','/'},
    {'!','?','@','#','$','%','&','*','(',')'},
    {' ','\b', 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 },  /* row 5: space, backspace, done */
};

char wubu_gn_kb_char(int row, int col) {
    if (row < 0 || row >= WUBU_GN_KB_ROWS || col < 0 || col >= WUBU_GN_KB_COLS)
        return 0;
    return g_kb[row][col];
}

int wubu_gn_init(wubu_gamepad_nav_t *gn) {
    if (!gn) return -1;
    memset(gn, 0, sizeof(*gn));
    gn->cx = WUBU_GN_SCREEN_W / 2;
    gn->cy = WUBU_GN_SCREEN_H / 2;
    gn->focus = -1;
    gn->speed = 8;
    gn->cursor_visible = 1;
    gn->kb_row = 1;   /* start on the QWERTY row */
    gn->kb_col = 0;
    return 0;
}

int wubu_gn_add_target(wubu_gamepad_nav_t *gn, int x, int y, const char *label) {
    if (!gn || !label) return -1;
    if (gn->n_targets >= WUBU_GN_MAX_TARGETS) return -1;
    wubu_gn_target_t *t = &gn->targets[gn->n_targets];
    t->x = x; t->y = y;
    strncpy(t->label, label, WUBU_GN_LABEL_MAX - 1);
    t->enabled = 1;
    return gn->n_targets++;
}

static int dist2(int ax, int ay, int bx, int by) {
    int dx = ax - bx, dy = ay - by;
    return dx * dx + dy * dy;
}

int wubu_gn_move_cursor(wubu_gamepad_nav_t *gn, int dx, int dy) {
    if (!gn) return -1;
    gn->cx += dx * gn->speed;
    gn->cy += dy * gn->speed;
    /* clamp to screen */
    if (gn->cx < 0) gn->cx = 0;
    if (gn->cx >= WUBU_GN_SCREEN_W) gn->cx = WUBU_GN_SCREEN_W - 1;
    if (gn->cy < 0) gn->cy = 0;
    if (gn->cy >= WUBU_GN_SCREEN_H) gn->cy = WUBU_GN_SCREEN_H - 1;

    /* snap to nearest enabled target within a 120px radius */
    int best = -1, best_d = 120 * 120;
    for (int i = 0; i < gn->n_targets; i++) {
        if (!gn->targets[i].enabled) continue;
        int d = dist2(gn->cx, gn->cy, gn->targets[i].x, gn->targets[i].y);
        if (d <= best_d) { best_d = d; best = i; }
    }
    gn->focus = best;
    return best;
}

int wubu_gn_set_focus(wubu_gamepad_nav_t *gn, int idx) {
    if (!gn || idx < 0 || idx >= gn->n_targets) return -1;
    if (!gn->targets[idx].enabled) return -1;
    gn->focus = idx;
    /* move the cursor onto the focused target */
    gn->cx = gn->targets[idx].x;
    gn->cy = gn->targets[idx].y;
    return 0;
}

int wubu_gn_confirm(wubu_gamepad_nav_t *gn) {
    if (!gn || gn->focus < 0 || gn->focus >= gn->n_targets) return -1;
    return 0;   /* focused target activated */
}

int wubu_gn_radial_open(wubu_gamepad_nav_t *gn) {
    if (!gn) return -1;
    gn->radial_open = 1;
    gn->radial_last = WUBU_GN_SECTOR_NONE;
    return 0;
}

int wubu_gn_radial_close(wubu_gamepad_nav_t *gn) {
    if (!gn) return -1;
    gn->radial_open = 0;
    return 0;
}

int wubu_gn_radial_select(wubu_gamepad_nav_t *gn, wubu_gn_sector_t sector) {
    if (!gn || !gn->radial_open) return -1;
    if (sector <= WUBU_GN_SECTOR_NONE || sector >= WUBU_GN_SECTOR_COUNT) return -1;
    gn->radial_last = sector;
    gn->n_radial_selects++;
    return 0;
}

int wubu_gn_kb_open(wubu_gamepad_nav_t *gn) {
    if (!gn) return -1;
    gn->kb_open = 1;
    gn->kb_len = 0;
    gn->kb_text[0] = '\0';
    gn->kb_row = 1;
    gn->kb_col = 0;
    return 0;
}

int wubu_gn_kb_close(wubu_gamepad_nav_t *gn) {
    if (!gn) return -1;
    gn->kb_open = 0;
    return 0;
}

int wubu_gn_kb_move(wubu_gamepad_nav_t *gn, int drow, int dcol) {
    if (!gn || !gn->kb_open) return -1;
    int r = gn->kb_row + drow, c = gn->kb_col + dcol;
    if (r < 0) r = 0;
    if (r >= WUBU_GN_KB_ROWS) r = WUBU_GN_KB_ROWS - 1;
    if (c < 0) c = 0;
    if (c >= WUBU_GN_KB_COLS) c = WUBU_GN_KB_COLS - 1;
    gn->kb_row = r;
    gn->kb_col = c;
    return 0;
}

int wubu_gn_kb_press(wubu_gamepad_nav_t *gn) {
    if (!gn || !gn->kb_open) return -1;
    char ch = wubu_gn_kb_char(gn->kb_row, gn->kb_col);
    if (gn->kb_row == WUBU_GN_KB_ROWS - 1) {
        if (gn->kb_col == 0) ch = ' ';        /* space */
        else if (gn->kb_col == 1) {           /* backspace */
            if (gn->kb_len > 0) gn->kb_text[--gn->kb_len] = '\0';
            return 0;
        }
        /* col 9 = done: close the keyboard */
        if (gn->kb_col == 9) { gn->kb_open = 0; return 0; }
    }
    if (ch == 0 || ch == '\b') return 0;     /* empty cell */
    if (gn->kb_len < (int)sizeof(gn->kb_text) - 1) {
        gn->kb_text[gn->kb_len++] = ch;
        gn->kb_text[gn->kb_len] = '\0';
    }
    return 0;
}

int wubu_gn_focus(const wubu_gamepad_nav_t *gn) {
    if (!gn) return -1;
    return gn->focus;
}

const char *wubu_gn_focus_label(const wubu_gamepad_nav_t *gn) {
    if (!gn || gn->focus < 0 || gn->focus >= gn->n_targets) return NULL;
    return gn->targets[gn->focus].label;
}

const char *wubu_gn_kb_text(const wubu_gamepad_nav_t *gn) {
    if (!gn) return NULL;
    return gn->kb_text;
}
