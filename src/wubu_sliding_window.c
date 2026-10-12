/* wubu_sliding_window.c — sliding-window attention mask (BRAIN #233).
 * C11, self-contained. Window radius w: |i-j| <= w is attended, else masked.
 */
#include "wubu_sliding_window.h"
#include <math.h>

int wubu_sw_allowed(int i, int j, int w) {
    if (w < 0) return 0;
    int d = i - j;
    if (d < 0) d = -d;
    return d <= w;
}

int wubu_sw_mask(int seq, int w, double *out_mask) {
    if (seq <= 0 || w < 0 || !out_mask) return -1;
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < seq; j++)
            out_mask[i * seq + j] = wubu_sw_allowed(i, j, w) ? 0.0 : -INFINITY;
    return 0;
}
