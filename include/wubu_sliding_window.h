#ifndef WUBU_SLIDING_WINDOW_H
#define WUBU_SLIDING_WINDOW_H
/* wubu_sliding_window.h — sliding-window attention mask (BRAIN #233).
 *
 * Each query attends only to keys within a fixed window of radius `w`
 * (|i - j| <= w); everything outside is masked (-inf). The forward
 * (wubu_sw_allowed) is covered by test_sliding_window with measured
 * assertions.
 */
#include <stddef.h>

/* Returns 1 if query i may attend to key j under window radius w, else 0. */
int wubu_sw_allowed(int i, int j, int w);

/* Fill a seq x seq mask (row-major, seq*seq doubles) with 0.0 (allowed) or
 * -INFINITY (masked) under window radius w. Returns 0 on ok, -1 on bad args. */
int wubu_sw_mask(int seq, int w, double *out_mask);

#endif
