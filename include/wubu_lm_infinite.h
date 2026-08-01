/*
 * wubu_lm_infinite.h -- Long-context memory mechanics (L13/O07/N20). Opaque where stateful.
 */
#ifndef WUBU_LM_INFINITE_H
#define WUBU_LM_INFINITE_H

#include <stddef.h>

/* L13 LM-Infinite landmark positions (every `stride`). */
int wubu_landmark_positions(int seq, int stride, int *out);

/* O07 Neuro sink positions (first n_sink tokens). */
int wubu_sink_positions(int seq, int n_sink, int *out);

/* N20 Shadow quant A/B state machine. */
typedef struct wubu_shadow wubu_shadow_t;
wubu_shadow_t *wubu_shadow_create(int ref_bits, int cheap_bits, int warm);
int wubu_shadow_observe(wubu_shadow_t *s, int matched);
void wubu_shadow_destroy(wubu_shadow_t *s);

#endif /* WUBU_LM_INFINITE_H */
