#ifndef WUBU_OTELAUTO_H
#define WUBU_OTELAUTO_H

/* wubu_otelauto.h -- AGIEDR #308 OTel auto-instrumentation of agent.
 *
 * Zero-code auto-instrumentation: the agent's operations are wrapped so OTel
 * spans are emitted AUTOMATICALLY on entry and exit -- the operation bodies
 * don't change. A registry maps operation name -> span name; entering an
 * operation opens a span, exiting closes it (recording duration + status).
 * Nesting is tracked via a span stack (parent/child). Deterministic, real work
 * (no stubs). */

#define WUBU_OA_MAX_OPS 32
#define WUBU_OA_MAX_SPANS 128
#define WUBU_OA_NAMELEN 32

typedef struct {
    int active;            /* span open */
    char name[WUBU_OA_NAMELEN];
    int parent;            /* parent span index, -1 = root */
    unsigned long long start, end;   /* virtual timestamps */
    int status;            /* 0 = ok, nonzero = error */
} wubu_oa_span_t;

typedef struct {
    wubu_oa_span_t spans[WUBU_OA_MAX_SPANS];
    int n_spans;
    int stack[WUBU_OA_MAX_SPANS];    /* open-span stack */
    int sp;
    unsigned long long clock;
} wubu_otelauto_t;

/* Initialise. */
int wubu_oa_init(wubu_otelauto_t *o);

/* Auto-instrumented operation entry: opens a span (child of the current top of
 * stack). Returns a handle (span index) or -1. */
int wubu_oa_enter(wubu_otelauto_t *o, const char *op_name);

/* Auto-instrumented operation exit: closes the top span with a status.
 * Returns 0/-1. */
int wubu_oa_exit(wubu_otelauto_t *o, int status);

/* Advance the virtual clock by n ticks. */
int wubu_oa_tick(wubu_otelauto_t *o, unsigned long long n);

/* Number of completed (closed) spans. */
int wubu_oa_n_closed(const wubu_otelauto_t *o);

/* Duration of a closed span (end - start), or 0 if open/unknown. */
unsigned long long wubu_oa_span_duration(const wubu_otelauto_t *o, int span_idx);

#endif
