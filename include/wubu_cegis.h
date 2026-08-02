/*
 * wubu_cegis.h -- Counterexample-guided inductive synthesis (EE03).
 */
#ifndef WUBU_CEGIS_H
#define WUBU_CEGIS_H

#define WUBU_CEGIS_MAX_CEX 32
#define WUBU_CEGIS_MAX_ITERS 100

typedef struct {
    int n_vars;
    /* Spec: f(x0, x1) must satisfy for all x in [lo_x, hi_x].
     * Here we target the "max" spec: f >= x0 && f >= x1 && (f==x0 || f==x1). */
    double lo, hi;
    int cex_x[WUBU_CEGIS_MAX_CEX];
    int cex_y[WUBU_CEGIS_MAX_CEX];
    int n_cex;
} wubu_cegis_spec_t;

typedef enum { CEGIS_CAND_MAX, CEGIS_CAND_ADD, CEGIS_CAND_SUB } wubu_cegis_cand_t;

typedef struct {
    wubu_cegis_cand_t kind;
    int found;   /* 1 if solution found */
} wubu_cegis_result_t;

/* Run CEGIS loop. Returns 1 if a valid candidate found (stored in *out). */
int wubu_cegis_run(wubu_cegis_spec_t *spec, unsigned seed, wubu_cegis_result_t *out);

/* Verify a candidate against the spec. Returns 1 if valid, 0 if counterexample
   found (writes to spec->cex). */
int wubu_cegis_verify(wubu_cegis_spec_t *spec, wubu_cegis_cand_t cand);

#endif