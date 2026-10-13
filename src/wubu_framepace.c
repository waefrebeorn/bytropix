/* wubu_framepace.c -- UX #248 Wayland frame pacing / vsync.
 * See wubu_framepace.h. */
#include "wubu_framepace.h"
#include <string.h>

int wubu_fp_init(wubu_framepace_t *p, uint64_t refresh_ns) {
    if (!p || refresh_ns == 0) return -1;
    memset(p->frames, 0, sizeof p->frames);
    p->n = 0;
    p->refresh_ns = refresh_ns;
    p->next_vsync = 0;
    p->n_missed = 0;
    return 0;
}

uint64_t wubu_fp_submit(wubu_framepace_t *p, uint64_t submit_time) {
    if (!p || p->n >= WUBU_FP_MAX_FRAMES) return 0;
    /* schedule to the next vsync tick at or after submit_time */
    uint64_t present = p->next_vsync;
    if (present < submit_time) {
        /* round up to the next vsync boundary */
        uint64_t overshoot = submit_time - present;
        uint64_t ticks = (overshoot + p->refresh_ns - 1) / p->refresh_ns;
        present += ticks * p->refresh_ns;
    }
    /* the frame missed its target if it was submitted after the vsync it
     * lands on would have been ready (i.e. it couldn't make the current tick) */
    int missed = (submit_time > p->next_vsync) ? 1 : 0;
    if (p->n > 0) {
        /* also late relative to the previous presentation + refresh */
        uint64_t prev = p->frames[p->n - 1].present_time;
        if (submit_time > prev + p->refresh_ns) missed = 1;
    }
    p->frames[p->n].present_time = present;
    p->frames[p->n].submit_time = submit_time;
    p->frames[p->n].missed = missed;
    if (missed) p->n_missed++;
    p->n++;
    p->next_vsync = present + p->refresh_ns;   /* next frame's earliest vsync */
    return present;
}

int wubu_fp_missed(const wubu_framepace_t *p) {
    return p ? p->n_missed : 0;
}

double wubu_fp_mean_interval(const wubu_framepace_t *p) {
    if (!p || p->n < 2) return 0.0;
    double sum = 0.0;
    for (int i = 1; i < p->n; i++)
        sum += (double)(p->frames[i].present_time - p->frames[i-1].present_time);
    return sum / (p->n - 1);
}

double wubu_fp_stability(const wubu_framepace_t *p) {
    if (!p || p->n < 2) return 1.0;
    double mean = wubu_fp_mean_interval(p);
    if (mean <= 0) return 1.0;
    double min_interval = 1e18;
    for (int i = 1; i < p->n; i++) {
        double iv = (double)(p->frames[i].present_time - p->frames[i-1].present_time);
        if (iv < min_interval) min_interval = iv;
    }
    double stab = min_interval / mean;
    if (stab > 1.0) stab = 1.0;
    return stab;
}
