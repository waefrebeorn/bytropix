/* test_shotgate.c -- STACK #367 tests: capture gates (a11y_shot + gui_shot). */
#include "wubu_shotgate.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

static wubu_sg_widget_t W(int x,int y,int w,int h,uint32_t fg,uint32_t bg,
                          const char *label,int interactive,int focused) {
    wubu_sg_widget_t r; memset(&r,0,sizeof(r));
    r.x=x;r.y=y;r.w=w;r.h=h;r.fg=fg;r.bg=bg;
    strncpy(r.label,label,WUBU_SG_LABEL_MAX-1);
    r.interactive=interactive; r.focused=focused; return r;
}

int main(void)
{
    wubu_sg_frame_t f;
    wubu_sg_result_t res;
    uint8_t fb[64];

    /* contrast helper sanity: black vs white = (255+5)/(0+5) = 260/5 = 52:1 */
    {
        int num, den;
        wubu_sg_contrast(0x000000, 0xffffff, &num, &den);
        CHECK(num == 260 && den == 5);
    }

    /* a fully-clean frame: 2 widgets, good contrast, one focused, no overlap */
    memset(&f, 0, sizeof(f));
    f.width = 800; f.height = 600; f.pixels = fb;
    f.min_contrast_num = 45; f.min_contrast_den = 10;   /* 4.5:1 */
    f.widgets[0] = W(10, 10, 100, 30, 0x000000, 0xffffff, "OK", 1, 1);      /* focused */
    f.widgets[1] = W(200, 10, 100, 30, 0xffffff, 0x000000, "Cancel", 1, 0);
    f.n_widgets = 2;
    CHECK(wubu_sg_evaluate(&f, &res) == 0);
    CHECK(res.a11y_pass == 1);
    CHECK(res.gui_pass == 1);
    CHECK(res.combined_pass == 1);
    CHECK(res.a11y_focus_count == 1);

    /* a11y failure: interactive widget missing a label */
    f.widgets[0].label[0] = '\0';
    CHECK(wubu_sg_evaluate(&f, &res) == 0);
    CHECK(res.a11y_missing_label == 1);
    CHECK(res.a11y_pass == 0);
    CHECK(res.combined_pass == 0);
    f.widgets[0] = W(10, 10, 100, 30, 0x000000, 0xffffff, "OK", 1, 1);   /* restore */

    /* a11y failure: low contrast (0x333333 on 0x222222, both dark) */
    f.widgets[1] = W(200, 10, 100, 30, 0x333333, 0x222222, "Cancel", 1, 0);
    CHECK(wubu_sg_evaluate(&f, &res) == 0);
    CHECK(res.a11y_low_contrast == 1);
    CHECK(res.a11y_pass == 0);
    f.widgets[1] = W(200, 10, 100, 30, 0xffffff, 0x000000, "Cancel", 1, 0);  /* restore */

    /* a11y failure: no focused widget */
    f.widgets[0].focused = 0;
    CHECK(wubu_sg_evaluate(&f, &res) == 0);
    CHECK(res.a11y_focus_count == 0);
    CHECK(res.a11y_pass == 0);
    /* a11y failure: two focused widgets */
    f.widgets[0].focused = 1; f.widgets[1].focused = 1;
    CHECK(wubu_sg_evaluate(&f, &res) == 0);
    CHECK(res.a11y_focus_count == 2);
    CHECK(res.a11y_pass == 0);
    f.widgets[1].focused = 0;   /* restore */

    /* gui failure: widget out of bounds */
    f.widgets[1] = W(750, 10, 100, 30, 0xffffff, 0x000000, "Cancel", 1, 0);  /* x+w=850 > 800 */
    CHECK(wubu_sg_evaluate(&f, &res) == 0);
    CHECK(res.gui_out_of_bounds == 1);
    CHECK(res.gui_pass == 0);
    CHECK(res.combined_pass == 0);
    f.widgets[1] = W(200, 10, 100, 30, 0xffffff, 0x000000, "Cancel", 1, 0);  /* restore */

    /* gui failure: overlapping widgets */
    f.widgets[1] = W(50, 10, 100, 30, 0xffffff, 0x000000, "Cancel", 1, 0);  /* overlaps w0 */
    CHECK(wubu_sg_evaluate(&f, &res) == 0);
    CHECK(res.gui_overlap == 1);
    CHECK(res.gui_pass == 0);
    f.widgets[1] = W(200, 10, 100, 30, 0xffffff, 0x000000, "Cancel", 1, 0);  /* restore */

    /* gui failure: empty frame */
    f.n_widgets = 0;
    CHECK(wubu_sg_evaluate(&f, &res) == 0);
    CHECK(res.gui_empty == 1);
    CHECK(res.gui_pass == 0);

    /* non-interactive widget with no label is fine (a11y only checks interactive) */
    f.n_widgets = 2;
    f.widgets[0] = W(10, 10, 100, 30, 0x000000, 0xffffff, "", 0, 1);   /* label, not interactive */
    f.widgets[1] = W(200, 10, 100, 30, 0xffffff, 0x000000, "Cancel", 1, 0);
    CHECK(wubu_sg_evaluate(&f, &res) == 0);
    CHECK(res.a11y_missing_label == 0);
    CHECK(res.a11y_pass == 1);

    /* bad args */
    CHECK(wubu_sg_evaluate(NULL, &res) == -1);
    CHECK(wubu_sg_evaluate(&f, NULL) == -1);

    if (fails == 0) printf("ALL SHOTGATE TESTS PASSED\n");
    else printf("%d SHOTGATE TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
