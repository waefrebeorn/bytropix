/*
 * wubu_invariant.h -- Loop invariant discovery (EE05).
 */
#ifndef WUBU_INVARIANT_H
#define WUBU_INVARIANT_H

#define WUBU_INV_MAX_INV 8
#define WUBU_INV_MAX_TRACE 64

typedef struct {
    /* A candidate loop invariant: a linear inequality over loop variables. */
    /* form: c0 + c1*x + c2*y >= 0  (x, y are loop vars, e.g. tok_s, iter) */
    double c0, c1, c2;
    char   desc[128];
} wubu_inv_t;

typedef struct {
    int n;             /* number of trace points */
    double x[WUBU_INV_MAX_TRACE];  /* var 1 (e.g. tok_s) */
    double y[WUBU_INV_MAX_TRACE];  /* var 2 (e.g. iter) */
} wubu_inv_trace_t;

typedef struct {
    wubu_inv_t invariants[WUBU_INV_MAX_INV];
    int n_inv;
} wubu_inv_set_t;

/* Given a trace of loop states, discover all candidate invariants that hold
   at every trace point. Returns number discovered. */
int wubu_invariant_discover(const wubu_inv_trace_t *trace, wubu_inv_set_t *out);

/* Verify an invariant holds on a new trace (inductive check). */
int wubu_invariant_check(const wubu_inv_t *inv, const wubu_inv_trace_t *trace);

#endif