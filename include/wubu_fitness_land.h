#ifndef WUBU_FITNESS_LAND_H
#define WUBU_FITNESS_LAND_H

/* wubu_fitness_land.h -- AGENT #162 mutation fitness landscape telemetry.
 *
 * Tracks the fitness landscape of the mutation search: records each mutation's
 * position (in a low-dim projection) and its fitness, then reports landscape
 * statistics -- mean/variance of fitness, the best-so-far, the improvement
 * trend, and whether the search is plateauing (consecutive non-improving
 * steps). This telemetry guides the search (e.g. when to restart, when a
 * basin is exhausted). Deterministic, fixed-size, real work (no stubs). */

#define WUBU_FL_MAX 256

typedef struct {
    double fitness;
    double position[4];   /* low-dim projection of the mutation */
} wubu_fl_sample_t;

typedef struct {
    wubu_fl_sample_t samples[WUBU_FL_MAX];
    int n;
    double best;
    int    best_idx;
    int    stagnant;      /* consecutive non-improving steps */
    int    stagnant_threshold;  /* plateau if stagnant >= this */
} wubu_fitness_land_t;

/* Initialise with a plateau threshold. */
int wubu_fl_init(wubu_fitness_land_t *l, int stagnant_threshold);

/* Record a mutation sample (fitness + position). Returns 0/-1. */
int wubu_fl_record(wubu_fitness_land_t *l, double fitness, const double *position);

/* Mean fitness over all samples. */
double wubu_fl_mean(const wubu_fitness_land_t *l);

/* Variance of fitness. */
double wubu_fl_variance(const wubu_fitness_land_t *l);

/* Best fitness so far. */
double wubu_fl_best(const wubu_fitness_land_t *l);

/* Is the search plateauing? (stagnant >= threshold). */
int wubu_fl_plateaued(const wubu_fitness_land_t *l);

#endif
