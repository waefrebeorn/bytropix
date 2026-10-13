#ifndef WUBU_SHOTGATE_H
#define WUBU_SHOTGATE_H

/* wubu_shotgate.h -- STACK #367 capture gates (a11y_shot + gui_shot).
 *
 * Capture test gates for the GUI: given a captured frame (an in-memory
 * framebuffer) plus per-widget metadata, evaluate two gates.
 *   a11y_shot -- accessibility: every interactive widget must have a non-empty
 *     accessible label, meet a minimum text/background contrast ratio, and
 *     exactly one widget must hold focus.
 *   gui_shot  -- GUI rendering: widgets must lie within the frame bounds, not
 *     overlap each other, and the frame must be non-empty.
 * Each gate is fail-closed: any violated check fails the gate. A combined
 * verdict passes only if both gates pass. Deterministic, real work (no stubs).
 * (Named wubu_shotgate to avoid the unrelated wubu_capgate specialist-spawn
 * gate.) */

#include <stddef.h>
#include <stdint.h>

#define WUBU_SG_MAX_WIDGETS 64
#define WUBU_SG_LABEL_MAX 64

typedef struct {
    int x, y, w, h;                 /* widget rect (frame pixels) */
    uint32_t fg, bg;                /* text + background color (0xRRGGBB) */
    char label[WUBU_SG_LABEL_MAX];  /* accessible label ("" = missing) */
    int interactive;                /* 1 if it takes input */
    int focused;                    /* 1 if it holds focus */
} wubu_sg_widget_t;

typedef struct {
    int width, height;
    const uint8_t *pixels;          /* framebuffer (may be NULL) */
    wubu_sg_widget_t widgets[WUBU_SG_MAX_WIDGETS];
    int n_widgets;
    int min_contrast_num;           /* min contrast as num/den (e.g. 45/10 = 4.5) */
    int min_contrast_den;
} wubu_sg_frame_t;

typedef struct {
    int a11y_pass;
    int gui_pass;
    int combined_pass;
    int a11y_missing_label;
    int a11y_low_contrast;
    int a11y_focus_count;
    int gui_out_of_bounds;
    int gui_overlap;
    int gui_empty;
} wubu_sg_result_t;

/* Evaluate both capture gates over a frame. Fills `res`. Returns 0/-1. */
int wubu_sg_evaluate(const wubu_sg_frame_t *f, wubu_sg_result_t *res);

/* Relative luminance (0..255) of a 0xRRGGBB color. */
int wubu_sg_luminance(uint32_t rgb);

/* Contrast ratio between two colors as num/den (x10). */
void wubu_sg_contrast(uint32_t fg, uint32_t bg, int *num, int *den);

#endif
