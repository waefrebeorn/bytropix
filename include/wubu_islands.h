#ifndef WUBU_ISLANDS_H
#define WUBU_ISLANDS_H

/* wubu_islands.h -- AGENT #163 island-model parallel evolution.
 *
 * Island-model evolution: the population is split into N independent
 * "islands" that evolve separately (preserving diversity), with periodic
 * MIGRATION of the best individuals between islands (sharing discoveries).
 * This balances exploration (islands diverge) and exploitation (migration
 * spreads good solutions). Deterministic, fixed-size, real work (no stubs). */

#define WUBU_ISL_MAX_ISLANDS 8
#define WUBU_ISL_POP 16

typedef struct {
    double fitness[WUBU_ISL_POP];
    int n;
    double best;
} wubu_isl_island_t;

typedef struct {
    wubu_isl_island_t islands[WUBU_ISL_MAX_ISLANDS];
    int n_islands;
    int migration_interval;   /* migrate every N generations */
    int generation;
    int n_migrations;
} wubu_islands_t;

/* Initialise with n islands and a migration interval. */
int wubu_isl_init(wubu_islands_t *m, int n_islands, int migration_interval);

/* Add an individual (fitness) to an island. Returns 0/-1. */
int wubu_isl_add(wubu_islands_t *m, int island, double fitness);

/* Best fitness on an island. */
double wubu_isl_island_best(const wubu_islands_t *m, int island);

/* Global best across all islands. */
double wubu_isl_global_best(const wubu_islands_t *m);

/* Advance a generation: if generation % interval == 0, migrate the best
 * individual of each island to the next island (ring topology). Returns the
 * number of migrations performed this generation. */
int wubu_isl_generation(wubu_islands_t *m);

/* Total migrations so far. */
int wubu_isl_migrations(const wubu_islands_t *m);

#endif
