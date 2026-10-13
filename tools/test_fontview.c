/* test_fontview.c -- Win98/XP Font viewer tests. */
#include "wubu_fontview.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_fontview_t fv;

    /* init: empty, default sample, preview 24 */
    CHECK(wubu_fv_init(&fv) == 0);
    CHECK(fv.n_fonts == 0);
    CHECK(fv.selected == -1);
    CHECK(fv.active == -1);
    CHECK(strcmp(fv.sample, "The quick brown fox") == 0);
    CHECK(fv.preview_size == 24);
    CHECK(wubu_fv_init(NULL) == -1);

    /* style names */
    CHECK(strcmp(wubu_fv_style_name(WUBU_FV_REGULAR), "Regular") == 0);
    CHECK(strcmp(wubu_fv_style_name(WUBU_FV_BOLD), "Bold") == 0);
    CHECK(strcmp(wubu_fv_style_name(WUBU_FV_ITALIC), "Italic") == 0);
    CHECK(strcmp(wubu_fv_style_name(WUBU_FV_BOLD_ITALIC), "Bold Italic") == 0);
    CHECK(wubu_fv_style_name((wubu_fv_style_t)99) == NULL);

    /* install fonts */
    CHECK(wubu_fv_install(&fv, "Arial", WUBU_FV_REGULAR, 10, 0) == 0);
    CHECK(wubu_fv_install(&fv, "Courier New", WUBU_FV_BOLD, 12, 1) == 1);
    CHECK(wubu_fv_install(&fv, "Times", WUBU_FV_ITALIC, 14, 0) == 2);
    CHECK(fv.n_fonts == 3);
    CHECK(fv.selected == 0);   /* first auto-selected */
    /* dup family+style allowed? my impl doesn't dedup; but test install variety */
    CHECK(wubu_fv_install(&fv, NULL, WUBU_FV_REGULAR, 10, 0) == -1);
    CHECK(wubu_fv_install(&fv, "Bad", WUBU_FV_REGULAR, 0, 0) == -1);   /* size 0 */
    CHECK(wubu_fv_install(&fv, "Bad", WUBU_FV_REGULAR, -1, 0) == -1);
    CHECK(wubu_fv_install(&fv, "Bad", (wubu_fv_style_t)99, 10, 0) == -1);
    CHECK(wubu_fv_install(NULL, "x", WUBU_FV_REGULAR, 10, 0) == -1);

    /* query fonts */
    CHECK(strcmp(wubu_fv_family(&fv, 0), "Arial") == 0);
    const wubu_fv_font_t *f = wubu_fv_font(&fv, 1);
    CHECK(f != NULL);
    CHECK(f->point_size == 12 && f->monospace == 1 && f->style == WUBU_FV_BOLD);
    CHECK(wubu_fv_font(&fv, 999) == NULL);
    CHECK(wubu_fv_family(&fv, 999) == NULL);
    CHECK(wubu_fv_font(NULL, 0) == NULL);

    /* select + active */
    CHECK(wubu_fv_select(&fv, 2) == 0);
    CHECK(wubu_fv_selected(&fv) == 2);
    CHECK(strcmp(wubu_fv_family(&fv, 2), "Times") == 0);
    CHECK(wubu_fv_set_active(&fv, 1) == 0);
    CHECK(wubu_fv_active(&fv) == 1);
    CHECK(strcmp(wubu_fv_family(&fv, 1), "Courier New") == 0);
    CHECK(wubu_fv_select(&fv, 999) == -1);
    CHECK(wubu_fv_select(&fv, -1) == -1);
    CHECK(wubu_fv_set_active(&fv, 999) == -1);
    CHECK(wubu_fv_select(NULL, 0) == -1);
    CHECK(wubu_fv_set_active(NULL, 0) == -1);

    /* sample + preview size */
    CHECK(wubu_fv_set_sample(&fv, "Hello World 123") == 0);
    CHECK(strcmp(fv.sample, "Hello World 123") == 0);
    CHECK(wubu_fv_set_sample(&fv, NULL) == -1);
    CHECK(wubu_fv_set_sample(NULL, "x") == -1);
    CHECK(wubu_fv_set_preview_size(&fv, 48) == 0);
    CHECK(fv.preview_size == 48);
    CHECK(wubu_fv_set_preview_size(&fv, 0) == -1);
    CHECK(wubu_fv_set_preview_size(&fv, -1) == -1);
    CHECK(wubu_fv_set_preview_size(NULL, 24) == -1);

    /* long sample truncates */
    char big[200];
    memset(big, 'x', sizeof(big) - 1); big[sizeof(big)-1] = '\0';
    CHECK(wubu_fv_set_sample(&fv, big) == 0);
    CHECK(strlen(fv.sample) == WUBU_FV_SAMPLE_MAX - 1);   /* truncated, NUL-terminated */

    /* selection survives selecting the active font */
    CHECK(wubu_fv_select(&fv, 0) == 0);
    CHECK(wubu_fv_set_active(&fv, 0) == 0);
    CHECK(wubu_fv_selected(&fv) == 0 && wubu_fv_active(&fv) == 0);

    /* bad args on query */
    CHECK(wubu_fv_selected(NULL) == -1);
    CHECK(wubu_fv_active(NULL) == -1);
    CHECK(wubu_fv_family(NULL, 0) == NULL);
    CHECK(wubu_fv_style_name(WUBU_FV_REGULAR) != NULL);

    if (fails == 0) printf("ALL FONTVIEW TESTS PASSED\n");
    else printf("%d FONTVIEW TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
