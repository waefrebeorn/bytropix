/* wubu_evosandbox.c -- PATHS #325 self-referential code evolution sandbox.
 * See wubu_evosandbox.h. */
#include "wubu_evosandbox.h"
#include <string.h>

int wubu_es_init(wubu_evosandbox_t *s, double start_fitness, int step_budget) {
    if (!s || step_budget < 0 || step_budget > WUBU_ES_MAX_MUTATIONS) return -1;
    memset(s, 0, sizeof *s);
    s->fitness = start_fitness;
    s->step_budget = step_budget;
    return 0;
}

int wubu_es_mutate(wubu_evosandbox_t *s, const char *name, double new_fitness, int safe) {
    if (!s || !name) return -1;
    if (s->step_budget <= 0) return -1;              /* budget exhausted */
    if (s->n >= WUBU_ES_MAX_MUTATIONS) return -1;
    if (strlen(name) >= WUBU_ES_NAMELEN) return -1;

    /* a mutation is kept only if it is SAFE and IMPROVES fitness */
    int improve = new_fitness > s->fitness;
    int kept = (safe && improve) ? 1 : 0;

    wubu_es_mutation_t *m = &s->muts[s->n];
    strncpy(m->name, name, WUBU_ES_NAMELEN - 1); m->name[WUBU_ES_NAMELEN-1] = '\0';
    m->fitness = new_fitness;
    m->safe = safe ? 1 : 0;
    m->kept = kept;
    s->n++;

    if (kept) {
        s->fitness = new_fitness;   /* apply */
        s->n_kept++;
    } else {
        s->n_reverted++;           /* revert: fitness unchanged */
    }
    s->step_budget--;              /* every attempt costs a step */
    return kept;
}

double wubu_es_fitness(const wubu_evosandbox_t *s) {
    return s ? s->fitness : 0.0;
}

int wubu_es_remaining(const wubu_evosandbox_t *s) {
    return s ? s->step_budget : 0;
}

void wubu_es_stats(const wubu_evosandbox_t *s, int *kept, int *reverted) {
    if (!s) return;
    if (kept) *kept = s->n_kept;
    if (reverted) *reverted = s->n_reverted;
}
