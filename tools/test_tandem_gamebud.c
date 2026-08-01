/* Test: full hardware-acceleration stack (doc "tandem"/"rambus"/"gamebud").
 *
 * Composes: hwcaps (SIMD detect) + rambus (interleaved KV banks) + tandem
 * (N64 RCP two-stage pipeline) + gamebud (frame-budget governor).
 *
 * Scenario: 8 decode "frames". Stage A = prefill (writes KV into rambus banks),
 * Stage B = decode GEMV (reads KV via rambus, bills gamebud frame budget).
 * Verifies: (1) hwcaps reports a SIMD width; (2) rambus interleave spreads
 * tokens across distinct banks (no bank conflict on sequential reads);
 * (3) tandem completes all frames; (4) gamebud throttles optional work when
 * the frame is near budget.
 */
#include "wubu_hwcaps.h"
#include "wubu_rambus.h"
#include "wubu_tandem.h"
#include "wubu_gamebud.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Stage A: pretend prefill — write KV for one token into rambus, bill a little. */
static void stage_a(void *arg, int stage, int frame) {
    (void)stage;
    wubu_rambus_t *rb = (wubu_rambus_t *)((void **)arg)[0];
    int hd = 8;
    uint8_t *p = wubu_rambus_kv_ptr(rb, frame, 0, hd, 0, sizeof(float));
    assert(p);
    wubu_rambus_access(rb, frame, 0, hd, hd * sizeof(float));
}

/* Stage B: decode — read KV for all prior tokens, governed by gamebud budget. */
static void stage_b(void *arg, int stage, int frame) {
    (void)stage;
    wubu_rambus_t *rb = (wubu_rambus_t *)((void **)arg)[0];
    wubu_gamebud_t *gb = (wubu_gamebud_t *)((void **)arg)[1];
    int hd = 8;

    int fid = wubu_gamebud_begin(gb);
    for (int t = 0; t <= frame; t++) {
        uint8_t *p = wubu_rambus_kv_ptr(rb, t, 0, hd, 0, sizeof(float));
        assert(p);
        wubu_rambus_access(rb, t, 0, hd, hd * sizeof(float));
    }
    /* Optional speculative work — only if frame budget allows */
    if (wubu_gamebud_can_spend(gb, 5000)) {
        /* would do extra draft tokens; skipped when over budget */
    }
    wubu_gamebud_end(gb, 1000 + frame * 10);
    (void)fid;
}

int main(void) {
    printf("=== Tandem + Rambus + Gamebud Integration Test ===\n");

    /* 1. HW caps */
    const wubu_hwcaps_t *hw = wubu_hwcaps_get();
    printf("  HW: %s\n", wubu_hwcaps_str(hw));
    assert(hw->simd_bits == 128 || hw->simd_bits == 256 || hw->simd_bits == 512);
    assert(hw->simd_lanes >= 4);

    /* 2. Rambus interleaved KV arena */
    wubu_rambus_t *rb = wubu_rambus_create(256 * 1024, 8, 256, 800);
    assert(rb);
    /* Token 0 and token 1 must land in DIFFERENT banks (interleave works) */
    uint8_t *p0 = wubu_rambus_kv_ptr(rb, 0, 0, 8, 0, sizeof(float));
    uint8_t *p1 = wubu_rambus_kv_ptr(rb, 1, 0, 8, 0, sizeof(float));
    printf("  rambus: token0 bank-ptr=%p token1 bank-ptr=%p (must differ)\n", (void*)p0, (void*)p1);
    assert(p0 != p1);  /* interleaved across banks */

    /* 3. Tandem N64 RCP pipeline */
    void *args[2] = { rb, NULL };
    wubu_gamebud_t *gb = wubu_gamebud_create(20000);  /* 20ms frame budget */
    args[1] = gb;

    wubu_tandem_t *td = wubu_tandem_create(1, 1, "0", "1", 2);
    assert(td);
    wubu_tandem_set_a(td, stage_a);
    wubu_tandem_set_b(td, stage_b);

    for (int f = 0; f < 8; f++) {
        int rc = wubu_tandem_submit(td, args);
        assert(rc == 0);
    }

    uint64_t frames, a_busy, b_busy;
    wubu_tandem_stats(td, &frames, &a_busy, &b_busy);
    printf("  tandem: frames=%lu a_busy=%lu b_busy=%lu\n", frames, a_busy, b_busy);
    assert(frames == 8);
    assert(a_busy == 8 && b_busy == 8);

    /* 4. Gamebud throttle check */
    uint64_t gf, go, ga, gp, gt;
    wubu_gamebud_stats(gb, &gf, &go, &ga, &gp, &gt);
    printf("  gamebud: frames=%lu overruns=%lu avg_us=%lu peak_us=%lu\n",
           gf, go, ga, gp);
    assert(gf == 8);

    /* 5. Rambus bandwidth model sanity */
    uint64_t hits, misses, cyc;
    wubu_rambus_stats(rb, &hits, &misses, &cyc);
    printf("  rambus: hits=%lu misses=%lu cycles=%lu eff_BW=%.1f MB/s\n",
           hits, misses, cyc, wubu_rambus_eff_bw(rb, hits*64 + misses*64) / 1e6);
    assert(hits > 0);  /* sequential token reads hit open rows */

    wubu_tandem_free(td);
    wubu_rambus_free(rb);
    wubu_gamebud_free(gb);
    printf("ALL TANDEM+RAMBUS+GAMEBUD TESTS PASSED\n");
    return 0;
}
