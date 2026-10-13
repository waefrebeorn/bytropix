/* wubu_otelauto.c -- AGIEDR #308 OTel auto-instrumentation of agent.
 * See wubu_otelauto.h. */
#include "wubu_otelauto.h"
#include <string.h>

int wubu_oa_init(wubu_otelauto_t *o) {
    if (!o) return -1;
    memset(o->spans, 0, sizeof o->spans);
    memset(o->stack, 0, sizeof o->stack);
    o->n_spans = 0;
    o->sp = 0;
    o->clock = 0;
    return 0;
}

int wubu_oa_tick(wubu_otelauto_t *o, unsigned long long n) {
    if (!o) return -1;
    o->clock += n;
    return 0;
}

int wubu_oa_enter(wubu_otelauto_t *o, const char *op_name) {
    if (!o || !op_name || o->n_spans >= WUBU_OA_MAX_SPANS || o->sp >= WUBU_OA_MAX_SPANS) return -1;
    if (strlen(op_name) >= WUBU_OA_NAMELEN) return -1;
    int idx = o->n_spans;
    wubu_oa_span_t *s = &o->spans[idx];
    s->active = 1;
    strncpy(s->name, op_name, WUBU_OA_NAMELEN - 1); s->name[WUBU_OA_NAMELEN-1] = '\0';
    s->parent = (o->sp > 0) ? o->stack[o->sp - 1] : -1;   /* child of current top */
    s->start = o->clock;
    s->end = 0;
    s->status = 0;
    o->stack[o->sp++] = idx;
    o->n_spans++;
    return idx;
}

int wubu_oa_exit(wubu_otelauto_t *o, int status) {
    if (!o || o->sp <= 0) return -1;
    int idx = o->stack[--o->sp];
    o->spans[idx].active = 0;
    o->spans[idx].end = o->clock;
    o->spans[idx].status = status;
    return 0;
}

int wubu_oa_n_closed(const wubu_otelauto_t *o) {
    if (!o) return 0;
    int c = 0;
    for (int i = 0; i < o->n_spans; i++) if (!o->spans[i].active) c++;
    return c;
}

unsigned long long wubu_oa_span_duration(const wubu_otelauto_t *o, int span_idx) {
    if (!o || span_idx < 0 || span_idx >= o->n_spans) return 0;
    const wubu_oa_span_t *s = &o->spans[span_idx];
    if (s->active) return 0;
    return s->end - s->start;
}
