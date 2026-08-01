/*
 * wubu_gamebud.h — Game-design frame-budget governor (doc "gamebud").
 *
 * Console games run at a fixed frame budget (e.g. 16.67 ms for 60 fps). If a
 * frame overruns, the engine drops non-essential work to hit the deadline.
 * We apply the same discipline to inference decode: each decode "frame" (one
 * token step) gets a hard time budget. Work that exceeds it (extra speculative
 * draft depth, optional quality passes) is THROTTLED so tail latency stays
 * bounded — exactly the fairness property the continuous-batching scheduler
 * needs (doc 007). This is the "game-design our inference" ask made concrete.
 *
 * Self-contained C11. No third-party deps.
 */
#ifndef WUBU_GAMEBUD_H
#define WUBU_GAMEBUD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_gamebud wubu_gamebud_t;

/* Create governor. budget_us = per-step time budget in microseconds
 * (e.g. 16667 for a 60fps-equivalent decode step). */
wubu_gamebud_t *wubu_gamebud_create(uint64_t budget_us);

void wubu_gamebud_free(wubu_gamebud_t *g);

/* Begin a frame. Returns a frame id. */
int wubu_gamebud_begin(wubu_gamebud_t *g);

/* Report that `us_used` microseconds were spent this frame (call once at end). */
void wubu_gamebud_end(wubu_gamebud_t *g, uint64_t us_used);

/* Query: given optional-work cost `us_optional`, should we run it this frame?
 * Returns 1 if it fits in the remaining budget, 0 if it would overrun. */
int wubu_gamebud_can_spend(wubu_gamebud_t *g, uint64_t us_optional);

/* Adaptive budget: if we keep under-running, the governor loans the saved time
 * to future frames (burst allowance). Returns current effective budget. */
uint64_t wubu_gamebud_effective_budget(const wubu_gamebud_t *g);

/* Returns microseconds elapsed since the last wubu_gamebud_begin() for this
 * governor (0 if no frame open). Lets the caller bill the real wall-clock. */
uint64_t wubu_gamebud_elapsed_us(const wubu_gamebud_t *g);

/* Stats: frames, overruns, avg/peak us, total throttled us. */
void wubu_gamebud_stats(const wubu_gamebud_t *g,
                        uint64_t *frames, uint64_t *overruns,
                        uint64_t *avg_us, uint64_t *peak_us,
                        uint64_t *throttled_us);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_GAMEBUD_H */
