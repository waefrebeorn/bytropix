/* test_sliding_window.c — BRAIN #233: sliding-window forward verified.
 *
 * Measured assertions on wubu_sw_allowed + wubu_sw_mask:
 *   1. self always allowed (i,i).
 *   2. within window allowed, outside masked (exact boundary w).
 *   3. symmetric.
 *   4. mask matches allowed(): 0.0 vs -inf, count of allowed == expected band.
 *   5. window 0 => only diagonal allowed.
 *   6. bad args rejected.
 */
#include "wubu_sliding_window.h"
#include <stdio.h>
#include <math.h>

static int g_run = 0, g_pass = 0;
#define T(cond, msg) do { \
    g_run++; \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while (0)

int main(void) {
    printf("=== test_sliding_window (sliding-window attention, #233) ===\n");

    const int w = 2;
    T(wubu_sw_allowed(5, 5, w) == 1, "self always allowed");

    /* 2: boundary. dist<=w allowed, dist>w masked. */
    T(wubu_sw_allowed(5, 7, w) == 1, "dist==w allowed");
    T(wubu_sw_allowed(5, 8, w) == 0, "dist==w+1 masked");
    T(wubu_sw_allowed(5, 3, w) == 1, "dist==-w allowed");
    T(wubu_sw_allowed(5, 2, w) == 0, "dist==-(w+1) masked");

    /* 3: symmetric. */
    T(wubu_sw_allowed(3, 9, w) == wubu_sw_allowed(9, 3, w), "symmetric");

    /* 4: mask matches allowed + band count. seq=6, w=2. */
    const int seq = 6;
    double m[36];
    T(wubu_sw_mask(seq, w, m) == 0, "mask built");
    int match = 1, allowed = 0;
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < seq; j++) {
            int a = wubu_sw_allowed(i, j, w);
            double v = m[i * seq + j];
            if (a && !(v == 0.0)) match = 0;
            if (!a && !isinf(v)) match = 0;
            if (a) allowed++;
        }
    T(match, "mask values match allowed()");
    /* Band count with edge clamping: for each row i, allowed keys are
     * j in [max(0,i-w), min(seq-1,i+w)] => count = min(seq-1,i+w)-max(0,i-w)+1. */
    int expected = 0;
    for (int i = 0; i < seq; i++) {
        int lo = i - w < 0 ? 0 : i - w;
        int hi = i + w > seq - 1 ? seq - 1 : i + w;
        expected += hi - lo + 1;
    }
    T(allowed == expected, "allowed count matches edge-clamped band");

    /* 5: window 0 => diagonal only. */
    T(wubu_sw_allowed(4, 4, 0) == 1 && wubu_sw_allowed(4, 5, 0) == 0, "w=0 diagonal only");

    /* 6: bad args. */
    T(wubu_sw_mask(0, w, m) == -1, "seq<=0 rejected");
    T(wubu_sw_mask(seq, -1, m) == -1, "w<0 rejected");
    T(wubu_sw_mask(seq, w, NULL) == -1, "NULL mask rejected");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}
