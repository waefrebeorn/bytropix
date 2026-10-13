/* test_gamepad_nav.c -- Steam gamepad virtual cursor + radial menu tests. */
#include "wubu_gamepad_nav.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_gamepad_nav_t gn;

    /* init: cursor centered, no focus, speed 8 */
    CHECK(wubu_gn_init(&gn) == 0);
    CHECK(gn.cx == 640 && gn.cy == 400);
    CHECK(gn.focus == -1);
    CHECK(gn.speed == 8);

    /* add targets */
    CHECK(wubu_gn_add_target(&gn, 100, 100, "Play") == 0);
    CHECK(wubu_gn_add_target(&gn, 1180, 100, "Store") == 1);
    CHECK(wubu_gn_add_target(&gn, 640, 700, "Library") == 2);
    CHECK(gn.n_targets == 3);
    CHECK(wubu_gn_add_target(&gn, 0, 0, NULL) == -1);
    CHECK(wubu_gn_add_target(NULL, 0, 0, "x") == -1);

    /* move cursor toward Play (100,100): from center (640,400) */
    CHECK(wubu_gn_move_cursor(&gn, -68, -38) == 0);   /* -> (96, 96) near Play */
    CHECK(wubu_gn_focus(&gn) == 0);
    CHECK(strcmp(wubu_gn_focus_label(&gn), "Play") == 0);

    /* cursor clamps to screen */
    CHECK(wubu_gn_move_cursor(&gn, -1000, -1000) == -1);  /* far from any target */
    CHECK(gn.cx == 0 && gn.cy == 0);

    /* move to Store (1180,100) */
    CHECK(wubu_gn_move_cursor(&gn, 147, 13) == 1);   /* -> (1176,104) near Store */
    CHECK(wubu_gn_focus(&gn) == 1);
    CHECK(strcmp(wubu_gn_focus_label(&gn), "Store") == 0);

    /* confirm requires a focus */
    CHECK(wubu_gn_confirm(&gn) == 0);                /* Store focused */
    /* navigate to Library (640,700) from Store (1176,104) */
    CHECK(wubu_gn_move_cursor(&gn, -67, 75) == 2);   /* -> (640,704) near Library */
    CHECK(wubu_gn_focus(&gn) == 2);
    CHECK(strcmp(wubu_gn_focus_label(&gn), "Library") == 0);
    /* clear focus (corner far from all targets) then confirm fails */
    CHECK(wubu_gn_move_cursor(&gn, 0, -2000) == -1);  /* -> (640,0) far from all */
    CHECK(wubu_gn_focus(&gn) == -1);
    CHECK(wubu_gn_confirm(&gn) == -1);
    CHECK(wubu_gn_focus_label(&gn) == NULL);   /* no focus -> NULL label */

    /* set_focus directly moves the cursor onto the target */
    CHECK(wubu_gn_set_focus(&gn, 0) == 0);
    CHECK(gn.focus == 0);
    CHECK(gn.cx == 100 && gn.cy == 100);
    CHECK(wubu_gn_set_focus(&gn, 99) == -1);   /* out of range */
    CHECK(wubu_gn_set_focus(NULL, 0) == -1);

    /* radial menu */
    CHECK(gn.radial_open == 0);
    CHECK(wubu_gn_radial_select(&gn, WUBU_GN_SECTOR_UP) == -1);  /* closed */
    CHECK(wubu_gn_radial_open(&gn) == 0);
    CHECK(gn.radial_open == 1);
    CHECK(wubu_gn_radial_select(&gn, WUBU_GN_SECTOR_UP) == 0);
    CHECK(gn.radial_last == WUBU_GN_SECTOR_UP);
    CHECK(wubu_gn_radial_select(&gn, WUBU_GN_SECTOR_RIGHT) == 0);
    CHECK(gn.n_radial_selects == 2);
    CHECK(wubu_gn_radial_select(&gn, WUBU_GN_SECTOR_NONE) == -1);  /* bad sector */
    CHECK(wubu_gn_radial_select(&gn, (wubu_gn_sector_t)99) == -1);
    CHECK(wubu_gn_radial_close(&gn) == 0);
    CHECK(gn.radial_open == 0);
    CHECK(wubu_gn_radial_select(&gn, WUBU_GN_SECTOR_UP) == -1);  /* closed again */

    /* virtual keyboard */
    CHECK(wubu_gn_kb_char(0, 0) == '1');
    CHECK(wubu_gn_kb_char(1, 0) == 'q');
    CHECK(wubu_gn_kb_char(2, 0) == 'a');
    CHECK(wubu_gn_kb_char(5, 0) == ' ');
    CHECK(wubu_gn_kb_char(5, 1) == '\b');
    CHECK(wubu_gn_kb_char(5, 9) == 0);      /* done marker */
    CHECK(wubu_gn_kb_char(99, 0) == 0);     /* out of range */

    CHECK(wubu_gn_kb_open(&gn) == 0);
    CHECK(gn.kb_open == 1);
    CHECK(gn.kb_row == 1 && gn.kb_col == 0);   /* QWERTY row start */
    /* type "hi": h is row2 col5, i is row1 col7 */
    CHECK(wubu_gn_kb_move(&gn, 1, 5) == 0);    /* -> row2 col5 = 'h' */
    CHECK(wubu_gn_kb_press(&gn) == 0);
    CHECK(strcmp(wubu_gn_kb_text(&gn), "h") == 0);
    CHECK(wubu_gn_kb_move(&gn, -1, 2) == 0);   /* -> row1 col7 = 'i' */
    CHECK(wubu_gn_kb_press(&gn) == 0);
    CHECK(strcmp(wubu_gn_kb_text(&gn), "hi") == 0);

    /* backspace: row5 col1. From row1 col7 -> (row5, col1) */
    CHECK(wubu_gn_kb_move(&gn, 4, -6) == 0);   /* -> row5 col1 = backspace */
    CHECK(wubu_gn_kb_press(&gn) == 0);
    CHECK(strcmp(wubu_gn_kb_text(&gn), "h") == 0);

    /* space via row5 col0: from row5 col1 -> col0 */
    CHECK(wubu_gn_kb_move(&gn, 0, -1) == 0);   /* -> row5 col0 = space */
    CHECK(wubu_gn_kb_press(&gn) == 0);
    CHECK(strcmp(wubu_gn_kb_text(&gn), "h ") == 0);

    /* navigation clamps at grid edges */
    CHECK(wubu_gn_kb_move(&gn, -100, -100) == 0);
    CHECK(gn.kb_row == 0 && gn.kb_col == 0);
    CHECK(wubu_gn_kb_move(&gn, 100, 100) == 0);
    CHECK(gn.kb_row == WUBU_GN_KB_ROWS-1 && gn.kb_col == WUBU_GN_KB_COLS-1);

    /* done key (row5 col9) closes the keyboard */
    CHECK(wubu_gn_kb_move(&gn, 0, -9) == 0);   /* -> row5 col0 */
    CHECK(wubu_gn_kb_move(&gn, 0, 9) == 0);    /* -> row5 col9 = done */
    CHECK(wubu_gn_kb_press(&gn) == 0);
    CHECK(gn.kb_open == 0);

    /* kb ops fail when closed */
    CHECK(wubu_gn_kb_move(&gn, 0, 1) == -1);
    CHECK(wubu_gn_kb_press(&gn) == -1);
    CHECK(wubu_gn_kb_close(&gn) == 0);

    /* text accessor + bad args */
    CHECK(wubu_gn_kb_text(&gn) != NULL);
    CHECK(wubu_gn_kb_text(NULL) == NULL);
    CHECK(wubu_gn_focus_label(NULL) == NULL);
    CHECK(wubu_gn_move_cursor(NULL, 1, 1) == -1);
    CHECK(wubu_gn_confirm(NULL) == -1);
    CHECK(wubu_gn_radial_open(NULL) == -1);
    CHECK(wubu_gn_init(NULL) == -1);

    if (fails == 0) printf("ALL GAMEPAD_NAV TESTS PASSED\n");
    else printf("%d GAMEPAD_NAV TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
