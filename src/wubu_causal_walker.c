/* wubu_causal_walker.c — AGENT #311: causal walker localizes a divergence.
 * C11, self-contained. Walks the causal chain backward to the root cause.
 */
#include "wubu_causal_walker.h"

int wubu_walker_init(wubu_walker_t *w, int n, double threshold) {
    if (!w || n < 0 || n > WUBU_WALKER_MAX_CELLS) return -1;
    w->n = n;
    w->threshold = threshold > 0 ? threshold : 3.0;
    for (int i = 0; i < WUBU_WALKER_MAX_CELLS; i++) w->z[i] = 0.0;
    return 0;
}

int wubu_walker_set_z(wubu_walker_t *w, int cell, double z) {
    if (!w || cell < 0 || cell >= w->n) return -1;
    w->z[cell] = z;
    return 0;
}

int wubu_walker_anomalous(const wubu_walker_t *w, int cell) {
    if (!w || cell < 0 || cell >= w->n) return -1;
    return w->z[cell] > w->threshold ? 1 : 0;
}

int wubu_walker_localize(const wubu_walker_t *w, int d) {
    if (!w || d < 0 || d >= w->n) return -1;
    /* walk FORWARD from 0 to d; the first (earliest / lowest-index)
     * anomalous cell is the root cause. A later divergence that is
     * anomalous itself but whose upstream is clean is downstream
     * propagation, not the source. */
    for (int c = 0; c <= d; c++)
        if (w->z[c] > w->threshold) return c;
    return -1;   /* no cell up to d is anomalous: unattributable */
}
