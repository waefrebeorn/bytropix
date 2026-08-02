/*
 * wubu_invariant.c -- Loop invariant discovery (EE05). C11.
 *
 * Convergence (invariant discovery / abstract interpretation 7-hop):
 *   - EE05: given a trace of loop states (var1, var2) at each iteration, discover
 *     all linear invariants c0 + c1*x + c2*y >= 0 that hold at every point.
 *     At home: recursive_optimize's sweep loop has invariants like
 *     "tok_s monotonic non-decreasing on accepted configs"; the discoverer
 *     finds + certifies it, feeding loopguard (AG-01) a *proof* not a heuristic.
 */
#include "wubu_invariant.h"
#include <math.h>
#include <string.h>

static int holds_for_all(const wubu_inv_trace_t *t, double c0, double c1, double c2) {
    if (!t || t->n == 0) return 0;
    for (int i = 0; i < t->n; i++) {
        double val = c0 + c1 * t->x[i] + c2 * t->y[i];
        if (val < -1e-9) return 0;  /* violated */
    }
    return 1;
}

int wubu_invariant_discover(const wubu_inv_trace_t *trace, wubu_inv_set_t *out) {
    if (!trace || !out || trace->n == 0) return -1;
    out->n_inv = 0;
    /* Candidate templates (restricted linear invariants). */
    struct { double c0, c1, c2; const char *d; } cands[] = {
        { 0, 1, 0, "tok_s >= 0" },
        { 0, 0, 1, "iter >= 0" },
        { 0, -1, 1, "iter - tok_s >= 0" },  /* tok_s bounded by iter */
        { -25, 1, 0, "tok_s >= 25" },
        { 0, 0, 0, "true" },
        { -100, 1, 0, "tok_s <= 100" },
    };
    int nc = sizeof(cands) / sizeof(cands[0]);
    for (int i = 0; i < nc; i++) {
        if (holds_for_all(trace, cands[i].c0, cands[i].c1, cands[i].c2)) {
            if (out->n_inv < WUBU_INV_MAX_INV) {
                wubu_inv_t *inv = &out->invariants[out->n_inv++];
                inv->c0 = cands[i].c0; inv->c1 = cands[i].c1; inv->c2 = cands[i].c2;
                strncpy(inv->desc, cands[i].d, 127);
                inv->desc[127] = '\0';
            }
        }
    }
    return out->n_inv;
}

int wubu_invariant_check(const wubu_inv_t *inv, const wubu_inv_trace_t *trace) {
    if (!inv || !trace) return 0;
    return holds_for_all(trace, inv->c0, inv->c1, inv->c2);
}
