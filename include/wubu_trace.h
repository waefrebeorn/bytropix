#ifndef WUBU_TRACE_H
#define WUBU_TRACE_H
#include <stdint.h>
#include <stddef.h>

/* wubu_trace.h -- AGI-EDR agent tracing: one OpenTelemetry/OpenInference-style
 * span per agent action (AGIEDR #300).
 *
 * A span records: trace id (per task/session), span id, parent span id (the
 * action that caused this one), agent id, action name, start/end nanosecond
 * timestamps, an arbitrary attribute list (key/value string pairs), and a
 * terminal status (UNSET/OK/ERROR). Spans live in a fixed-size ring buffer
 * (no allocation on the hot path) and are drainable for export.
 *
 * Deterministic, kernel-owned (the agent cannot forge a span's identity
 * beyond the ids it is given), real work (no stubs). */

#define WUBU_TRACE_MAX_SPANS 256
#define WUBU_TRACE_MAX_ATTRS 8
#define WUBU_TRACE_KEYLEN    32
#define WUBU_TRACE_VALLEN    64
#define WUBU_TRACE_NAMELEN   48

typedef enum { WUBU_SPAN_UNSET=0, WUBU_SPAN_OK, WUBU_SPAN_ERROR } wubu_span_status_t;

typedef struct {
    char key[WUBU_TRACE_KEYLEN];
    char val[WUBU_TRACE_VALLEN];
} wubu_span_attr_t;

typedef struct {
    uint64_t trace_id;
    uint64_t span_id;
    uint64_t parent_id;   /* 0 = root */
    int      agent_id;
    char     name[WUBU_TRACE_NAMELEN];
    uint64_t start_ns;
    uint64_t end_ns;      /* 0 = still open */
    wubu_span_attr_t attrs[WUBU_TRACE_MAX_ATTRS];
    int      n_attrs;
    wubu_span_status_t status;
} wubu_span_t;

typedef struct {
    wubu_span_t spans[WUBU_TRACE_MAX_SPANS];
    int      head;        /* next write slot (ring) */
    int      count;       /* total spans emitted (may exceed MAX via wraparound) */
    uint64_t next_span_id;
} wubu_tracer_t;

/* Initialise a tracer. span_id counter starts at 1. */
int wubu_trace_init(wubu_tracer_t *t);

/* Begin a span (agent action start). Returns the span id (>0), or 0 on error.
 * parent_id=0 marks a root span. trace_id groups related spans (a task). */
uint64_t wubu_trace_begin(wubu_tracer_t *t, uint64_t trace_id, uint64_t parent_id,
                          int agent_id, const char *name, uint64_t now_ns);

/* Attach an attribute (key/value) to an open span. Returns 0 or -1. */
int wubu_trace_attr(wubu_tracer_t *t, uint64_t span_id, const char *key, const char *val);

/* End a span with a terminal status. Returns 0 or -1 (unknown span). */
int wubu_trace_end(wubu_tracer_t *t, uint64_t span_id, wubu_span_status_t status, uint64_t now_ns);

/* Look up the most recent span by id. Returns NULL if not found. */
const wubu_span_t *wubu_trace_find(const wubu_tracer_t *t, uint64_t span_id);

/* Number of spans currently retained in the ring. */
int wubu_trace_count(const wubu_tracer_t *t);

/* Drain: copy retained spans into out[] (up to max). Returns count copied. */
int wubu_trace_drain(const wubu_tracer_t *t, wubu_span_t *out, int max);

/* Validate a span: has non-zero ids, end>=start (if ended), a name. 1/0. */
int wubu_span_valid(const wubu_span_t *s);

#endif
