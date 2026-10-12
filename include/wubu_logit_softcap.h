#ifndef WUBU_LOGIT_SOFTCAP_H
#define WUBU_LOGIT_SOFTCAP_H
/* wubu_logit_softcap.h — logit soft-capping (BRAIN #239).
 *
 * Gemma-2 style softcap: capped(x) = cap * tanh(x / cap). Bounds logits to
 * (-cap, cap) while staying ~linear near 0 (tanh(z)~z). Stabilizes training
 * without a hard clip. The forward (wubu_softcap) is covered by
 * test_logit_softcap with measured assertions.
 */
#include <stddef.h>

/* Apply softcap to a single logit. Returns cap*tanh(x/cap). */
double wubu_softcap(double x, double cap);

/* Apply softcap in place to `n` logits. Returns 0 ok, -1 bad args. */
int wubu_softcap_arr(float *logits, int n, double cap);

#endif
