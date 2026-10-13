/* wubu_shotgate.c -- STACK #367 capture gates (a11y_shot + gui_shot).
 * See wubu_shotgate.h. Real, deterministic capture-gate engine. */

#include "wubu_shotgate.h"
#include <string.h>

int wubu_sg_luminance(uint32_t rgb)
{
    int r = (rgb >> 16) & 0xff, g = (rgb >> 8) & 0xff, b = rgb & 0xff;
    /* Rec. 709 luma, scaled to 0..255 */
    return (r * 54 + g * 183 + b * 19) / 256;
}

void wubu_sg_contrast(uint32_t fg, uint32_t bg, int *num, int *den)
{
    int lf = wubu_sg_luminance(fg);
    int lb = wubu_sg_luminance(bg);
    int hi = lf > lb ? lf : lb;
    int lo = lf > lb ? lb : lf;
    /* WCAG contrast ratio = (L_hi + 5) / (L_lo + 5), luminance in 0..255.
     * Returned as num/den (both small ints); compare against a min ratio
     * min_num/min_den via num*min_den >= min_num*den. */
    *num = hi + 5;
    *den = lo + 5;
}

static int rects_overlap(int ax,int ay,int aw,int ah,int bx,int by,int bw,int bh)
{
    return (ax < bx + bw) && (bx < ax + aw) && (ay < by + bh) && (by < ay + ah);
}

int wubu_sg_evaluate(const wubu_sg_frame_t *f, wubu_sg_result_t *res)
{
    if (!f || !res) return -1;
    memset(res, 0, sizeof(*res));

    /* ---- a11y_shot ---- */
    int a11y_ok = 1;
    int focus = 0;
    for (int i = 0; i < f->n_widgets; i++) {
        const wubu_sg_widget_t *w = &f->widgets[i];
        if (w->focused) focus++;
        if (w->interactive) {
            if (w->label[0] == '\0') { res->a11y_missing_label++; a11y_ok = 0; }
            int num, den;
            wubu_sg_contrast(w->fg, w->bg, &num, &den);
            /* contrast (num/den) >= min (min_num/min_den) ?  num*min_den >= min_num*den */
            if ((int64_t)num * f->min_contrast_den < (int64_t)f->min_contrast_num * den) {
                res->a11y_low_contrast++; a11y_ok = 0;
            }
        }
    }
    res->a11y_focus_count = focus;
    if (focus != 1) a11y_ok = 0;   /* exactly one focused widget */
    res->a11y_pass = a11y_ok;

    /* ---- gui_shot ---- */
    int gui_ok = 1;
    if (f->n_widgets == 0) { res->gui_empty = 1; gui_ok = 0; }
    for (int i = 0; i < f->n_widgets; i++) {
        const wubu_sg_widget_t *w = &f->widgets[i];
        if (w->x < 0 || w->y < 0 || w->x + w->w > f->width || w->y + w->h > f->height ||
            w->w <= 0 || w->h <= 0) { res->gui_out_of_bounds++; gui_ok = 0; }
        for (int j = i + 1; j < f->n_widgets; j++) {
            const wubu_sg_widget_t *o = &f->widgets[j];
            if (rects_overlap(w->x,w->y,w->w,w->h, o->x,o->y,o->w,o->h)) {
                res->gui_overlap++; gui_ok = 0;
            }
        }
    }
    res->gui_pass = gui_ok;
    res->combined_pass = (res->a11y_pass && res->gui_pass) ? 1 : 0;
    return 0;
}
