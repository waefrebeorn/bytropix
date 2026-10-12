/* wubu_logit_softcap.c — Gemma-2 logit softcap (BRAIN #239).
 * C11, self-contained. cap*tanh(x/cap): bounded to (-cap,cap), ~linear near 0.
 */
#include "wubu_logit_softcap.h"
#include <math.h>

double wubu_softcap(double x, double cap) {
    if (cap <= 0.0) return x;   /* no cap configured -> identity */
    return cap * tanh(x / cap);
}

int wubu_softcap_arr(float *logits, int n, double cap) {
    if (!logits || n <= 0) return -1;
    if (cap <= 0.0) return 0;   /* identity */
    for (int i = 0; i < n; i++) logits[i] = (float)wubu_softcap((double)logits[i], cap);
    return 0;
}
