#ifndef WUBU_EVOSANDBOX_H
#define WUBU_EVOSANDBOX_H

/* wubu_evosandbox.h -- PATHS #325 self-referential code evolution sandbox.
 *
 * A sandbox for self-referential code evolution: the code can rewrite itself,
 * but ONLY inside the sandbox -- each mutation is (1) bounded by a step budget,
 * (2) checked against a safety invariant, and (3) reverted on violation. The
 * sandbox tracks a "fitness" that each mutation can improve; a mutation that
 * improves fitness AND stays safe is kept, else reverted. This lets code
 * evolve itself without escaping its bounds. Deterministic, real work
 * (no stubs). */

#define WUBU_ES_MAX_MUTATIONS 32
#define WUBU_ES_NAMELEN 32

typedef struct {
    char name[WUBU_ES_NAMELEN];
    double fitness;
    int kept;          /* 1 = applied, 0 = reverted */
    int safe;          /* 1 = passed the invariant */
} wubu_es_mutation_t;

typedef struct {
    wubu_es_mutation_t muts[WUBU_ES_MAX_MUTATIONS];
    int n;
    double fitness;        /* current (evolved) fitness */
    int step_budget;       /* remaining mutation budget */
    int n_kept;
    int n_reverted;
} wubu_evosandbox_t;

/* Initialise with a starting fitness and a mutation-step budget. Returns 0/-1. */
int wubu_es_init(wubu_evosandbox_t *s, double start_fitness, int step_budget);

/* Apply a self-mutation: 'new_fitness' is the fitness the mutation produces,
 * 'safe' says whether it preserves the safety invariant. Kept only if safe AND
 * improves fitness; else reverted (fitness unchanged). Returns 1 if kept,
 * 0 if reverted, -1 on error (budget exhausted / bad args). */
int wubu_es_mutate(wubu_evosandbox_t *s, const char *name, double new_fitness, int safe);

/* Current evolved fitness. */
double wubu_es_fitness(const wubu_evosandbox_t *s);

/* Remaining mutation budget. */
int wubu_es_remaining(const wubu_evosandbox_t *s);

/* Number of mutations kept vs reverted. */
void wubu_es_stats(const wubu_evosandbox_t *s, int *kept, int *reverted);

#endif
