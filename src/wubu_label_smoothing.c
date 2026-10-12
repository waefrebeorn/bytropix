/* wubu_label_smoothing.c — label smoothing (BRAIN #241).
 * C11, self-contained. target[y] = 1 - alpha + alpha/V; others = alpha/V.
 */
#include "wubu_label_smoothing.h"
#include <math.h>

int wubu_label_smooth(float *target, int V, int y, double alpha) {
    if (!target || V <= 0 || y < 0 || y >= V) return -1;
    if (alpha < 0.0 || alpha > 1.0) return -1;
    double off = alpha / (double)V;
    for (int k = 0; k < V; k++) target[k] = (float)off;
    target[y] = (float)(1.0 - alpha + off);
    return 0;
}

double wubu_label_smooth_loss(const float *target, const float *logp, int V) {
    if (!target || !logp || V <= 0) return -1.0;
    double loss = 0.0;
    for (int k = 0; k < V; k++) loss -= (double)target[k] * (double)logp[k];
    return loss;
}
