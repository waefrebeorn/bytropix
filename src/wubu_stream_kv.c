/*
 * wubu_stream_kv.c -- StreamingLLM attention-sink KV remapping (L01).
 *
 * Convergence (Kevin-Bacon 7-hop, StreamingLLM 2309.17453 + the I/O survey
 * 2026): at long context decode is KV-bandwidth/capacity bound. StreamingLLM
 * keeps the FIRST `sink` tokens (attention sinks) + a rolling window of the
 * most recent `window` tokens, and evicts the middle. This bounds KV cache
 * size to (sink + window) REGARDLESS of true sequence length -> infinite
 * streaming without OOM, and reuses the existing KV accessor array unchanged.
 *
 * This module owns NO tensor data. It only maps a *logical* token position
 * (0..L-1) to a *physical* KV-slot index inside the bounded cache, so the
 * existing kv_cache_read_elem(matmul) calls keep working. It is opt-in: when
 * disabled (default) the remap is the identity and behaviour is unchanged
 * (full 512K cache, no regression to the airllm budget path).
 *
 * Triple-DA:
 *  - Correctness: the live set is exactly {0..sink-1} U {L-window..L-1};
 *    remap is monotonic and surjective onto [0, sink+window).
 *  - Privacy: no external deps, no telemetry. Own C11 only.
 *  - Robustness: cap==0, window>=L (full cache), sink>L (all sink) handled.
 */
#include "wubu_stream_kv.h"
#include <stdlib.h>
#include <string.h>

struct wubu_stream_kv {
    int sink;       /* number of leading sink tokens kept */
    int window;     /* rolling window of recent tokens kept */
    int enabled;    /* 0 = identity remap (off) */
};

wubu_stream_kv_t *wubu_stream_kv_create(int sink, int window) {
    wubu_stream_kv_t *s = (wubu_stream_kv_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->sink   = sink   > 0 ? sink   : 0;
    s->window = window > 0 ? window : 0;
    s->enabled = (sink > 0 || window > 0) ? 1 : 0;
    return s;
}

void wubu_stream_kv_destroy(wubu_stream_kv_t *s) { free(s); }

void wubu_stream_kv_set(wubu_stream_kv_t *s, int sink, int window) {
    if (!s) return;
    s->sink   = sink   > 0 ? sink   : 0;
    s->window = window > 0 ? window : 0;
    s->enabled = (sink > 0 || window > 0) ? 1 : 0;
}

int wubu_stream_kv_enabled(const wubu_stream_kv_t *s) {
    return s ? s->enabled : 0;
}

int wubu_stream_kv_sink(const wubu_stream_kv_t *s) {
    return s ? s->sink : 0;
}

int wubu_stream_kv_window(const wubu_stream_kv_t *s) {
    return s ? s->window : 0;
}

/* Number of physical KV slots the bounded cache needs for a sequence of
 * `len` tokens. (sink + window) capped at len; 0 => off => identity (len). */
int wubu_stream_kv_capacity(const wubu_stream_kv_t *s, int len) {
    if (!s || !s->enabled || len <= 0) return len;
    int cap = s->sink + s->window;
    if (cap > len) cap = len;          /* never more than the real length */
    return cap;
}

/* Remap a logical position `pos` (0..len-1) to a physical slot, OR return
 * -1 if that token has been evicted (outside sink + window). When disabled,
 * returns `pos` unchanged (identity). */
int wubu_stream_kv_remap(const wubu_stream_kv_t *s, int len, int pos) {
    if (!s || !s->enabled) return pos;          /* identity: full cache */
    if (pos < 0 || pos >= len) return -1;

    int cap = wubu_stream_kv_capacity(s, len);
    if (cap >= len) return pos;                /* window covers everything */

    /* pos is in the sink region -> physical slot == pos */
    if (pos < s->sink) return pos;

    int recent_start = len - s->window;        /* first token in the window */
    if (pos >= recent_start) {
        /* map the window [recent_start, len-1] to [sink, cap-1] */
        return s->sink + (pos - recent_start);
    }
    /* evicted middle */
    return -1;
}

/* Count of live (non-evicted) tokens for a sequence of `len`. */
int wubu_stream_kv_live_count(const wubu_stream_kv_t *s, int len) {
    if (!s || !s->enabled || len <= 0) return len;
    int cap = wubu_stream_kv_capacity(s, len);
    if (cap >= len) return len;
    /* sink tokens + window tokens, but avoid double-counting if they overlap */
    int live = 0;
    for (int p = 0; p < len; p++)
        if (wubu_stream_kv_remap(s, len, p) >= 0) live++;
    return live;
}
