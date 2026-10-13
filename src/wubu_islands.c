/* wubu_islands.c -- AGENT #163 island-model parallel evolution.
 * See wubu_islands.h. */
#include "wubu_islands.h"
#include <string.h>

int wubu_isl_init(wubu_islands_t *m, int n_islands, int migration_interval) {
    if (!m || n_islands <= 0 || n_islands > WUBU_ISL_MAX_ISLANDS) return -1;
    memset(m->islands, 0, sizeof m->islands);
    m->n_islands = n_islands;
    m->migration_interval = migration_interval > 0 ? migration_interval : 5;
    m->generation = 0;
    m->n_migrations = 0;
    return 0;
}

int wubu_isl_add(wubu_islands_t *m, int island, double fitness) {
    if (!m || island < 0 || island >= m->n_islands) return -1;
    wubu_isl_island_t *is = &m->islands[island];
    if (is->n >= WUBU_ISL_POP) return -1;
    is->fitness[is->n++] = fitness;
    if (fitness > is->best) is->best = fitness;
    return 0;
}

double wubu_isl_island_best(const wubu_islands_t *m, int island) {
    if (!m || island < 0 || island >= m->n_islands) return 0.0;
    return m->islands[island].best;
}

double wubu_isl_global_best(const wubu_islands_t *m) {
    if (!m) return 0.0;
    double best = -1e18;
    for (int i = 0; i < m->n_islands; i++)
        if (m->islands[i].best > best) best = m->islands[i].best;
    return best < -1e17 ? 0.0 : best;
}

int wubu_isl_generation(wubu_islands_t *m) {
    if (!m) return -1;
    m->generation++;
    int migrated = 0;
    if (m->generation % m->migration_interval == 0) {
        /* ring migration: each island's best moves to the next island */
        for (int i = 0; i < m->n_islands; i++) {
            int next = (i + 1) % m->n_islands;
            double best = m->islands[i].best;
            if (m->islands[next].n < WUBU_ISL_POP && best > -1e17) {
                m->islands[next].fitness[m->islands[next].n++] = best;
                if (best > m->islands[next].best) m->islands[next].best = best;
                migrated++;
                m->n_migrations++;
            }
        }
    }
    return migrated;
}

int wubu_isl_migrations(const wubu_islands_t *m) {
    return m ? m->n_migrations : 0;
}
