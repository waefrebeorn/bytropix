/*
 * wubu_gamebud.c — Game-design frame-budget governor (doc "gamebud"). C11.
 */
#include "wubu_gamebud.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}
#else
static uint64_t now_us(void) { return 0; }
#endif

struct wubu_gamebud {
    uint64_t budget_us;
    uint64_t credit_us;       /* borrowed time from under-runs (burst) */
    uint64_t frame_start;     /* monotonic us at begin */
    int      in_frame;

    uint64_t frames, overruns;
    uint64_t sum_us, peak_us, throttled_us;
    int      last_frame;
};

wubu_gamebud_t *wubu_gamebud_create(uint64_t budget_us) {
    wubu_gamebud_t *g = (wubu_gamebud_t *)calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->budget_us = budget_us > 0 ? budget_us : 16667;
    g->credit_us = 0;
    return g;
}

void wubu_gamebud_free(wubu_gamebud_t *g) { free(g); }

int wubu_gamebud_begin(wubu_gamebud_t *g) {
    if (!g) return -1;
    g->frame_start = now_us();
    g->in_frame = 1;
    return g->last_frame++;
}

void wubu_gamebud_end(wubu_gamebud_t *g, uint64_t us_used) {
    if (!g || !g->in_frame) return;
    g->in_frame = 0;
    g->frames++;
    g->sum_us += us_used;
    if (us_used > g->peak_us) g->peak_us = us_used;

    uint64_t eff = wubu_gamebud_effective_budget(g);
    if (us_used > eff) {
        g->overruns++;
        /* over budget: drop credit, do NOT accumulate negative */
        g->credit_us = 0;
    } else {
        /* under budget: bank the savings as burst credit (capped) */
        uint64_t saved = eff - us_used;
        g->credit_us += saved;
        if (g->credit_us > g->budget_us * 4) g->credit_us = g->budget_us * 4;
    }
}

int wubu_gamebud_can_spend(wubu_gamebud_t *g, uint64_t us_optional) {
    if (!g) return 0;
    uint64_t eff = wubu_gamebud_effective_budget(g);
    uint64_t used = g->in_frame ? (now_us() - g->frame_start) : 0;
    if (used + us_optional <= eff) return 1;
    g->throttled_us += us_optional;
    return 0;
}

uint64_t wubu_gamebud_effective_budget(const wubu_gamebud_t *g) {
    if (!g) return 0;
    uint64_t eff = g->budget_us + g->credit_us;
    return eff;
}

void wubu_gamebud_stats(const wubu_gamebud_t *g,
                        uint64_t *frames, uint64_t *overruns,
                        uint64_t *avg_us, uint64_t *peak_us,
                        uint64_t *throttled_us) {
    if (frames)     *frames = g ? g->frames : 0;
    if (overruns)   *overruns = g ? g->overruns : 0;
    if (avg_us)     *avg_us = g && g->frames ? g->sum_us / g->frames : 0;
    if (peak_us)    *peak_us = g ? g->peak_us : 0;
    if (throttled_us)*throttled_us = g ? g->throttled_us : 0;
}
