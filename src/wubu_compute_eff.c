/* wubu_compute_eff.c -- PATHS #322 compute-efficiency: architectural wins.
 * See wubu_compute_eff.h. */
#include "wubu_compute_eff.h"
#include <string.h>

int wubu_ceff_init(wubu_compute_eff_t *c) {
    if (!c) return -1;
    memset(c->t, 0, sizeof c->t);
    c->n = 0;
    return 0;
}

int wubu_ceff_add(wubu_compute_eff_t *c, const char *name, double compute_mult, double quality_delta) {
    if (!c || !name || c->n >= 16) return -1;
    if (compute_mult <= 0.0) return -1;
    if (strlen(name) >= 40) return -1;
    strncpy(c->t[c->n].name, name, 39);
    c->t[c->n].compute_mult = compute_mult;
    c->t[c->n].quality_delta = quality_delta;
    c->n++;
    return 0;
}

double wubu_ceff_score(const wubu_eff_technique_t *t) {
    if (!t || t->compute_mult <= 0.0) return 0.0;
    return t->quality_delta / t->compute_mult;
}

int wubu_ceff_is_win(const wubu_eff_technique_t *t) {
    if (!t) return 0;
    return (t->compute_mult < 1.0 && t->quality_delta >= 0.0) ? 1 : 0;
}

int wubu_ceff_best(const wubu_compute_eff_t *c) {
    if (!c || c->n == 0) return -1;
    int best = 0;
    double best_score = wubu_ceff_score(&c->t[0]);
    for (int i = 1; i < c->n; i++) {
        double s = wubu_ceff_score(&c->t[i]);
        if (s > best_score) { best_score = s; best = i; }
    }
    return best;
}
