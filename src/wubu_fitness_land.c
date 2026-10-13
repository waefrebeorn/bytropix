/* wubu_fitness_land.c -- AGENT #162 mutation fitness landscape telemetry.
 * See wubu_fitness_land.h. */
#include "wubu_fitness_land.h"
#include <string.h>

int wubu_fl_init(wubu_fitness_land_t *l, int stagnant_threshold) {
    if (!l) return -1;
    memset(l->samples, 0, sizeof l->samples);
    l->n = 0;
    l->best = -1e18;
    l->best_idx = -1;
    l->stagnant = 0;
    l->stagnant_threshold = stagnant_threshold > 0 ? stagnant_threshold : 5;
    return 0;
}

int wubu_fl_record(wubu_fitness_land_t *l, double fitness, const double *position) {
    if (!l || l->n >= WUBU_FL_MAX) return -1;
    wubu_fl_sample_t *s = &l->samples[l->n];
    s->fitness = fitness;
    if (position) {
        for (int i = 0; i < 4; i++) s->position[i] = position[i];
    } else {
        for (int i = 0; i < 4; i++) s->position[i] = 0.0;
    }
    if (fitness > l->best) {
        l->best = fitness;
        l->best_idx = l->n;
        l->stagnant = 0;   /* improvement resets stagnation */
    } else {
        l->stagnant++;     /* no improvement */
    }
    l->n++;
    return 0;
}

double wubu_fl_mean(const wubu_fitness_land_t *l) {
    if (!l || l->n == 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < l->n; i++) sum += l->samples[i].fitness;
    return sum / l->n;
}

double wubu_fl_variance(const wubu_fitness_land_t *l) {
    if (!l || l->n == 0) return 0.0;
    double mean = wubu_fl_mean(l);
    double var = 0.0;
    for (int i = 0; i < l->n; i++) {
        double d = l->samples[i].fitness - mean;
        var += d * d;
    }
    return var / l->n;
}

double wubu_fl_best(const wubu_fitness_land_t *l) {
    if (!l || l->n == 0) return 0.0;
    return l->best;
}

int wubu_fl_plateaued(const wubu_fitness_land_t *l) {
    if (!l) return 0;
    return l->stagnant >= l->stagnant_threshold ? 1 : 0;
}
