/* test_wordproc.c -- Win98/XP Word processor tests. */
#include "wubu_wordproc.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_wordproc_t wp;
    char buf[256];

    CHECK(wubu_wp_init(&wp) == 0);
    CHECK(wubu_wp_len(&wp) == 0);
    CHECK(wubu_wp_cursor(&wp) == 0);
    CHECK(wubu_wp_dirty(&wp) == 0);
    CHECK(wubu_wp_init(NULL) == -1);

    /* insert plain text */
    CHECK(wubu_wp_insert(&wp, "hello world") == 11);
    CHECK(wubu_wp_len(&wp) == 11);
    CHECK(wubu_wp_cursor(&wp) == 11);
    CHECK(wubu_wp_dirty(&wp) == 1);
    CHECK(wubu_wp_words(&wp) == 2);
    CHECK(wubu_wp_text(&wp, buf, sizeof(buf)) == 11);
    CHECK(strcmp(buf, "hello world") == 0);

    /* insert at a cursor position (middle) */
    CHECK(wubu_wp_move_cursor(&wp, 5) == 0);
    CHECK(wubu_wp_insert(&wp, " there") == 6);
    CHECK(wubu_wp_text(&wp, buf, sizeof(buf)) == 17);
    CHECK(strcmp(buf, "hello there world") == 0);
    CHECK(wubu_wp_words(&wp) == 3);

    /* cursor clamp */
    CHECK(wubu_wp_move_cursor(&wp, 999) == 0);
    CHECK(wubu_wp_cursor(&wp) == 17);
    CHECK(wubu_wp_move_cursor(&wp, -5) == 0);
    CHECK(wubu_wp_cursor(&wp) == 0);
    CHECK(wubu_wp_move_cursor(NULL, 0) == -1);

    /* delete forward */
    CHECK(wubu_wp_move_cursor(&wp, 0) == 0);
    CHECK(wubu_wp_delete(&wp, 6) == 6);   /* delete "hello " */
    CHECK(wubu_wp_text(&wp, buf, sizeof(buf)) == 11);
    CHECK(strcmp(buf, "there world") == 0);
    CHECK(wubu_wp_delete(&wp, 999) == 11);   /* delete rest */
    CHECK(wubu_wp_len(&wp) == 0);

    /* selection + delete (replace) */
    CHECK(wubu_wp_insert(&wp, "abcdef") == 6);
    CHECK(wubu_wp_select(&wp, 1, 4) == 0);   /* select "bcd" */
    CHECK(wubu_wp_cursor(&wp) == 4);
    CHECK(wubu_wp_delete(&wp, 0) == 3);      /* delete selection */
    CHECK(wubu_wp_text(&wp, buf, sizeof(buf)) == 3);
    CHECK(strcmp(buf, "aef") == 0);

    /* insert with selection replaces it */
    CHECK(wubu_wp_insert(&wp, "XY") == 2);   /* cursor was 1: "aXYef" */
    CHECK(wubu_wp_select(&wp, 0, 3) == 0);   /* select "aXY" */
    CHECK(wubu_wp_insert(&wp, "Z") == 1);    /* replaces "aXY" with "Z" -> "Zef" */
    CHECK(wubu_wp_text(&wp, buf, sizeof(buf)) == 3);
    CHECK(strcmp(buf, "Zef") == 0);

    /* style: pending style applies to next insert */
    CHECK(wubu_wp_select(&wp, -1, -1) == 0);   /* clear selection */
    CHECK(wubu_wp_move_cursor(&wp, 3) == 0);
    CHECK(wubu_wp_toggle_style(&wp, WUBU_WP_BOLD) == 0);   /* pending bold */
    CHECK(wubu_wp_insert(&wp, "B") == 1);   /* "ZXYB", B is bold */
    CHECK(wubu_wp_style_at(&wp, 3) == WUBU_WP_BOLD);
    CHECK(wubu_wp_style_at(&wp, 2) == 0);   /* Y is not bold */

    /* toggle style on a selection */
    CHECK(wubu_wp_select(&wp, 0, 3) == 0);   /* select "ZXY" */
    CHECK(wubu_wp_toggle_style(&wp, WUBU_WP_ITALIC) == 0);
    CHECK(wubu_wp_style_at(&wp, 0) == WUBU_WP_ITALIC);
    CHECK(wubu_wp_style_at(&wp, 1) == WUBU_WP_ITALIC);
    CHECK(wubu_wp_style_at(&wp, 2) == WUBU_WP_ITALIC);
    /* toggle again removes it */
    CHECK(wubu_wp_toggle_style(&wp, WUBU_WP_ITALIC) == 0);
    CHECK(wubu_wp_style_at(&wp, 0) == 0);
    /* combine bold + italic on a range */
    CHECK(wubu_wp_toggle_style(&wp, WUBU_WP_BOLD) == 0);
    CHECK(wubu_wp_toggle_style(&wp, WUBU_WP_ITALIC) == 0);
    CHECK(wubu_wp_style_at(&wp, 0) == (WUBU_WP_BOLD | WUBU_WP_ITALIC));

    /* bad style bit */
    CHECK(wubu_wp_toggle_style(&wp, 0xFF) == -1);
    CHECK(wubu_wp_toggle_style(NULL, WUBU_WP_BOLD) == -1);

    /* mark_saved clears dirty */
    CHECK(wubu_wp_dirty(&wp) == 1);
    CHECK(wubu_wp_mark_saved(&wp) == 0);
    CHECK(wubu_wp_dirty(&wp) == 0);
    CHECK(wubu_wp_mark_saved(NULL) == -1);

    /* overflow guard: fill to exactly MAX with 1-char inserts */
    wubu_wordproc_t big;
    CHECK(wubu_wp_init(&big) == 0);
    int total = 0;
    for (int i = 0; i < WUBU_WP_MAX_TEXT + 10; i++) {
        int r = wubu_wp_insert(&big, "x");
        if (r < 0) break;
        total += r;
    }
    CHECK(total == WUBU_WP_MAX_TEXT);   /* filled exactly, no overflow */
    CHECK(wubu_wp_insert(&big, "x") == -1);   /* now full */
    CHECK(wubu_wp_len(&big) == WUBU_WP_MAX_TEXT);

    /* style_at out of range */
    CHECK(wubu_wp_style_at(&wp, 999) == 0);
    CHECK(wubu_wp_style_at(&wp, -1) == 0);

    /* bad args */
    CHECK(wubu_wp_insert(NULL, "x") == -1);
    CHECK(wubu_wp_insert(&wp, NULL) == -1);
    CHECK(wubu_wp_delete(NULL, 1) == -1);
    CHECK(wubu_wp_delete(&wp, -1) == -1);
    CHECK(wubu_wp_select(NULL, 0, 1) == -1);
    CHECK(wubu_wp_len(NULL) == -1);
    CHECK(wubu_wp_cursor(NULL) == -1);
    CHECK(wubu_wp_words(NULL) == -1);
    CHECK(wubu_wp_dirty(NULL) == -1);
    CHECK(wubu_wp_style_at(NULL, 0) == 0);
    CHECK(wubu_wp_text(&wp, NULL, 10) == -1);
    CHECK(wubu_wp_text(&wp, buf, 0) == -1);

    if (fails == 0) printf("ALL WORDPROC TESTS PASSED\n");
    else printf("%d WORDPROC TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
