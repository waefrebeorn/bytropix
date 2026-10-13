/* test_framepace.c -- UX #248 Wayland frame pacing / vsync.
 * Proves: submit schedules to vsync, on-time frames -> perfect stability,
 * late frame -> missed, mean interval, fail-closed on NULL. */
#include "wubu_framepace.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)
#define NEAR(a,b,t) (fabs((a)-(b)) < (t))

int main(void) {
    wubu_framepace_t p;
    uint64_t refresh = 16666667;   /* ~60Hz in ns */
    CHECK(wubu_fp_init(&p, refresh) == 0, "init ok (60Hz)");
    CHECK(wubu_fp_init(&p, 0) == -1, "zero refresh -> -1");

    /* submit a frame at t=0 -> presented on the first vsync (t=0) */
    uint64_t pt = wubu_fp_submit(&p, 0);
    CHECK(pt == 0, "frame at t=0 -> presented at vsync 0");
    CHECK(wubu_fp_missed(&p) == 0, "no missed yet");

    /* on-time frames: submit each just before its vsync (refresh - 1) */
    for (int i = 1; i < 6; i++)
        wubu_fp_submit(&p, (uint64_t)i * refresh - 1);
    CHECK(p.n == 6, "6 frames submitted");
    CHECK(wubu_fp_missed(&p) == 0, "0 missed (all on-time)");

    /* mean interval = refresh (perfect pacing) */
    CHECK(NEAR(wubu_fp_mean_interval(&p), (double)refresh, 1.0), "mean interval = refresh");
    /* stability = 1.0 (perfectly regular) */
    CHECK(NEAR(wubu_fp_stability(&p), 1.0, 1e-6), "stability = 1.0 (perfect)");

    /* a LATE frame: submit way past the next vsync -> missed */
    uint64_t late = 6 * refresh + refresh / 2;   /* half a frame late */
    wubu_fp_submit(&p, late);
    CHECK(wubu_fp_missed(&p) >= 1, "late frame -> missed");

    /* mean interval still ~ refresh on average */
    CHECK(wubu_fp_mean_interval(&p) > 0, "mean interval positive");

    /* fail-closed */
    CHECK(wubu_fp_init(NULL, refresh) == -1, "NULL init -> -1");
    CHECK(wubu_fp_submit(NULL, 0) == 0, "NULL submit -> 0");
    CHECK(wubu_fp_missed(NULL) == 0, "NULL missed -> 0");
    CHECK(wubu_fp_mean_interval(NULL) == 0.0, "NULL mean -> 0");

    printf("\n=== test_framepace: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
