/*
 * wubu_lm_infinite.c -- Long-context memory mechanics (L13 / O07 / N20). C11.
 *
 * Convergence (LM-Infinite landmark 2305.14398 + Titans neuro sink + online
 * shadow-A/B 7-hop):
 *  - L13 LM-Infinite: landmark ("soft prompt") tokens are injected every `stride`
 *    positions; they summarize the segment between them so the model can attend
 *    the landmark instead of the full window (unbounded context, fixed memory).
 *    This module computes which positions are landmarks and the segment each
 *    landmark covers.
 *  - O07 Neuro sink neurons: attention-sink KEEP -- certain positions (the first
 *    `n_sink` tokens) are *always* retained as global memory anchors (ties L01
 *    StreamingKV sink). Returns the set of sink positions.
 *  - N20 Shadow quant A/B: run two KV precisions in shadow; switch to the cheaper
 *    one once it has matched the reference quality for `warm` steps. Pure state
 *    machine; returns the currently-active bits.
 *
 * Triple-DA: stride<=0 / n_sink<0 / warm<=0 clamped; deterministic.
 */
#include "wubu_lm_infinite.h"
#include <stdlib.h>
#include <string.h>

struct wubu_shadow {
    int ref_bits;
    int cheap_bits;
    int warm;
    int active;
    int match_streak;
    int switched;
};

/* L13 LM-Infinite landmark positions: every `stride` tokens is a landmark
 * (position stride, 2*stride, ...). Writes landmark positions into out (caller
 * sized >= seq/stride). Returns count. stride<=0 -> 0. */
int wubu_landmark_positions(int seq, int stride, int *out) {
    if (seq <= 0 || stride <= 0 || !out) return 0;
    int c = 0;
    for (int p = stride; p < seq; p += stride) out[c++] = p;
    return c;
}

/* O07 Neuro sink positions: the first n_sink tokens are permanent sinks.
 * Writes them into out, returns count (clamped to seq). */
int wubu_sink_positions(int seq, int n_sink, int *out) {
    if (seq <= 0 || !out) return 0;
    if (n_sink < 0) n_sink = 0;
    if (n_sink > seq) n_sink = seq;
    for (int i = 0; i < n_sink; i++) out[i] = i;
    return n_sink;
}

/* N20 Shadow quant A/B state machine. Create with the two candidate bit-widths
 * (ref = high-quality reference, cheap = the one we want to switch to). */
wubu_shadow_t *wubu_shadow_create(int ref_bits, int cheap_bits, int warm) {
    if (ref_bits <= 0 || cheap_bits <= 0) return NULL;
    if (warm <= 0) warm = 1;
    wubu_shadow_t *s = (wubu_shadow_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->ref_bits = ref_bits; s->cheap_bits = cheap_bits; s->warm = warm;
    s->active = ref_bits; s->match_streak = 0; s->switched = 0;
    return s;
}

/* Feed one comparison result (1 = cheap matched reference, 0 = diverged).
 * Returns the currently-active bits. Once `warm` consecutive matches accrue,
 * switches active to cheap_bits permanently. */
int wubu_shadow_observe(wubu_shadow_t *s, int matched) {
    if (!s) return 0;
    if (s->switched) return s->active;
    if (matched) {
        s->match_streak++;
        if (s->match_streak >= s->warm) {
            s->active = s->cheap_bits;
            s->switched = 1;
        }
    } else {
        s->match_streak = 0;
    }
    return s->active;
}

void wubu_shadow_destroy(wubu_shadow_t *s) { free(s); }
