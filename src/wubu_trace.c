/* wubu_trace.c -- AGI-EDR agent tracing: OTel/OpenInference span per action.
 * See wubu_trace.h. Fixed-size ring buffer, no allocation on the hot path. */
#include "wubu_trace.h"
#include <string.h>

int wubu_trace_init(wubu_tracer_t *t) {
    if (!t) return -1;
    memset(t, 0, sizeof(*t));
    t->head = 0;
    t->count = 0;
    t->next_span_id = 1;
    return 0;
}

static wubu_span_t *slot_for(wubu_tracer_t *t, uint64_t span_id) {
    for (int i = 0; i < WUBU_TRACE_MAX_SPANS; i++) {
        int idx = (t->head - 1 - i + 2*WUBU_TRACE_MAX_SPANS) % WUBU_TRACE_MAX_SPANS;
        if (t->spans[idx].span_id == span_id) return &t->spans[idx];
    }
    return NULL;
}

uint64_t wubu_trace_begin(wubu_tracer_t *t, uint64_t trace_id, uint64_t parent_id,
                          int agent_id, const char *name, uint64_t now_ns) {
    if (!t || !name) return 0;
    uint64_t sid = t->next_span_id++;
    wubu_span_t *s = &t->spans[t->head];
    memset(s, 0, sizeof(*s));
    s->trace_id = trace_id ? trace_id : sid;   /* default trace = own span */
    s->span_id = sid;
    s->parent_id = parent_id;
    s->agent_id = agent_id;
    strncpy(s->name, name, WUBU_TRACE_NAMELEN - 1);
    s->start_ns = now_ns;
    s->end_ns = 0;
    s->status = WUBU_SPAN_UNSET;
    s->n_attrs = 0;
    t->head = (t->head + 1) % WUBU_TRACE_MAX_SPANS;
    t->count++;
    return sid;
}

int wubu_trace_attr(wubu_tracer_t *t, uint64_t span_id, const char *key, const char *val) {
    if (!t || !key || !val) return -1;
    wubu_span_t *s = slot_for(t, span_id);
    if (!s) return -1;
    if (s->n_attrs >= WUBU_TRACE_MAX_ATTRS) return -1;
    wubu_span_attr_t *a = &s->attrs[s->n_attrs++];
    strncpy(a->key, key, WUBU_TRACE_KEYLEN - 1);
    strncpy(a->val, val, WUBU_TRACE_VALLEN - 1);
    return 0;
}

int wubu_trace_end(wubu_tracer_t *t, uint64_t span_id, wubu_span_status_t status, uint64_t now_ns) {
    if (!t) return -1;
    wubu_span_t *s = slot_for(t, span_id);
    if (!s) return -1;
    if (s->end_ns != 0) return -1;            /* already ended */
    if (now_ns < s->start_ns) now_ns = s->start_ns;  /* monotonic clamp */
    s->end_ns = now_ns;
    s->status = status;
    return 0;
}

const wubu_span_t *wubu_trace_find(const wubu_tracer_t *t, uint64_t span_id) {
    if (!t) return NULL;
    for (int i = 0; i < WUBU_TRACE_MAX_SPANS; i++) {
        int idx = (t->head - 1 - i + 2*WUBU_TRACE_MAX_SPANS) % WUBU_TRACE_MAX_SPANS;
        if (t->spans[idx].span_id == span_id) return &t->spans[idx];
    }
    return NULL;
}

int wubu_trace_count(const wubu_tracer_t *t) {
    if (!t) return 0;
    return t->count < WUBU_TRACE_MAX_SPANS ? t->count : WUBU_TRACE_MAX_SPANS;
}

int wubu_trace_drain(const wubu_tracer_t *t, wubu_span_t *out, int max) {
    if (!t || !out || max <= 0) return 0;
    int n = wubu_trace_count(t);
    if (n > max) n = max;
    /* copy oldest-first */
    int start = (t->head - wubu_trace_count(t) + 2*WUBU_TRACE_MAX_SPANS) % WUBU_TRACE_MAX_SPANS;
    for (int i = 0; i < n; i++) {
        out[i] = t->spans[(start + i) % WUBU_TRACE_MAX_SPANS];
    }
    return n;
}

int wubu_span_valid(const wubu_span_t *s) {
    if (!s) return 0;
    if (s->span_id == 0 || s->trace_id == 0) return 0;
    if (s->name[0] == '\0') return 0;
    if (s->end_ns != 0 && s->end_ns < s->start_ns) return 0;
    return 1;
}
