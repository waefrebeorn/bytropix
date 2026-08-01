/*
 * wubu_causal.h -- Causal + neuro-symbolic substrate for AGI-OS (AW01-AW10).
 */
#ifndef WUBU_CAUSAL_H
#define WUBU_CAUSAL_H

#include <stddef.h>

#define WUBU_SCM_MAX 32
#define WUBU_SCM_MAX_EDGES 64
#define WUBU_ABDUCT_MAX 16

/* AW01-AW04: Structural Causal Model. */
typedef struct {
    int   n;                       /* number of variables */
    double val[WUBU_SCM_MAX];      /* current assignment */
    int   edges[WUBU_SCM_MAX_EDGES][2];
    int   n_edges;
} wubu_scm_t;

int  wubu_scm_add_edge(wubu_scm_t *m, int cause, int effect);
int  wubu_scm_do(const wubu_scm_t *m, int a, double val, double *out);
int  wubu_scm_counterfactual(const wubu_scm_t *m, int a, double val,
                                 int target, double *out_cf);
int  wubu_scm_identifiable(const wubu_scm_t *m, int x, int y);

/* AW06(belief): temporal belief revision. */
double wubu_belief_update(double prior, double lik);

/* AW06/AW09/AW10: abductive diagnosis. */
typedef struct {
    double prior;          /* prior probability of this cause */
    double likelihood;     /* P(obs | cause) */
    int    explains[WUBU_SCM_MAX];  /* which observations this explains */
} wubu_abduct_t;

int  wubu_abduct_best(const wubu_abduct_t *ax, int n, int obs,
                          int *out_best, double *out_score);
int  wubu_counter_abduct(const wubu_abduct_t *a, const wubu_abduct_t *b);

/* AW08: PDDL-lite planner (proposition-bitmask states). */
typedef struct {
    int   n_prop;
    int   n_actions;
    unsigned *precond;   /* [n_actions x nwords] */
    unsigned *effect;    /* [n_actions x nwords] (xor mask) */
} wubu_pddl_t;

int  wubu_pddl_plan(const wubu_pddl_t *p, const unsigned *init, const unsigned *goal,
                        int max_steps, int *out_actions);

#endif
