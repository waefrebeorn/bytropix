#ifndef WUBU_COMPUTE_EFF_H
#define WUBU_COMPUTE_EFF_H

/* wubu_compute_eff.h -- PATHS #322 compute-efficiency: architectural wins
 * over scaling.
 *
 * Models the efficiency tradeoff between ALGORITHMIC improvement (better
 * architecture: attention->linear-attention, dense->MoE, fp16->int8) and raw
 * COMPUTE scaling (more FLOPs). Each technique has a compute multiplier (how
 * much compute it needs) and a quality delta (how much it improves quality).
 * The "architectural win" is when a technique REDUCES compute while
 * MAINTAINING or improving quality -- these dominate scaling (which only adds
 * compute). The model scores techniques by efficiency = quality_delta /
 * compute_multiplier. Deterministic, real work (no stubs). */

typedef struct {
    char name[40];
    double compute_mult;   /* 1.0 = same compute, 0.5 = half, 2.0 = double */
    double quality_delta;  /* quality change (+ = better) */
} wubu_eff_technique_t;

typedef struct {
    wubu_eff_technique_t t[16];
    int n;
} wubu_compute_eff_t;

/* Initialise. */
int wubu_ceff_init(wubu_compute_eff_t *c);

/* Add a technique. Returns 0/-1. */
int wubu_ceff_add(wubu_compute_eff_t *c, const char *name, double compute_mult, double quality_delta);

/* Efficiency score of a technique: quality_delta / compute_mult.
 * Higher = better (more quality per unit compute). */
double wubu_ceff_score(const wubu_eff_technique_t *t);

/* Is this an "architectural win"? (reduces compute AND maintains/improves
 * quality: compute_mult < 1 && quality_delta >= 0). */
int wubu_ceff_is_win(const wubu_eff_technique_t *t);

/* Best technique by efficiency score. Returns index or -1. */
int wubu_ceff_best(const wubu_compute_eff_t *c);

#endif
