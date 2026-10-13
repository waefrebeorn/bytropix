#ifndef WUBU_FRAMEPACE_H
#define WUBU_FRAMEPACE_H
#include <stdint.h>

/* wubu_framepace.h -- UX #248 Wayland frame pacing / vsync.
 *
 * Frame pacing: the compositor presents frames at the display's refresh
 * interval (vsync). This pacer schedules each frame to the next vsync tick and
 * reports presentation timing -- the presentation timestamp, whether the frame
 * MISSED its target (late), and the frame-to-frame jitter. Feedback lets the
 * client pace its rendering to the display. Deterministic, real work
 * (no stubs). */

#define WUBU_FP_MAX_FRAMES 64

typedef struct {
    uint64_t present_time;   /* the vsync tick this frame was presented on */
    uint64_t submit_time;    /* when the frame was submitted */
    int missed;              /* 1 = frame missed its target vsync (late) */
} wubu_fp_frame_t;

typedef struct {
    wubu_fp_frame_t frames[WUBU_FP_MAX_FRAMES];
    int n;
    uint64_t refresh_ns;     /* vsync interval in ns (e.g. 16.67ms = 16666667) */
    uint64_t next_vsync;     /* next vsync tick */
    int n_missed;
} wubu_framepace_t;

/* Initialise with the refresh interval (ns per frame). Returns 0/-1. */
int wubu_fp_init(wubu_framepace_t *p, uint64_t refresh_ns);

/* Submit a frame at submit_time (ns). It is scheduled to the next vsync tick.
 * Returns the presentation timestamp, or 0 on error. */
uint64_t wubu_fp_submit(wubu_framepace_t *p, uint64_t submit_time);

/* Number of missed (late) frames. */
int wubu_fp_missed(const wubu_framepace_t *p);

/* Mean frame interval (ns) between consecutive presentations. Returns 0 if
 * fewer than 2 frames. */
double wubu_fp_mean_interval(const wubu_framepace_t *p);

/* Frame pacing stability: 1.0 = perfectly on-interval, lower = jittery.
 * Computed as min/mean of the inter-frame intervals (in (0,1]). */
double wubu_fp_stability(const wubu_framepace_t *p);

#endif
