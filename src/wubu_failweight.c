/* wubu_failweight.c -- AGENT #148 DA-2 failure weighting 3x bounds search.
 * See wubu_failweight.h. */
#include "wubu_failweight.h"

int wubu_failweight_init(wubu_failweight_t *f, int n_regions, int failure_weight) {
    if (!f || n_regions <= 0 || n_regions > WUBU_FAILWEIGHT_MAX_REGIONS) return -1;
    for (int i = 0; i < WUBU_FAILWEIGHT_MAX_REGIONS; i++) {
        f->regions[i].successes = 0;
        f->regions[i].failures = 0;
    }
    f->n_regions = n_regions;
    f->failure_weight = failure_weight > 0 ? failure_weight : 3;
    return 0;
}

int wubu_failweight_record(wubu_failweight_t *f, int region, int success) {
    if (!f || region < 0 || region >= f->n_regions) return -1;
    if (success) f->regions[region].successes++;
    else f->regions[region].failures++;
    return 0;
}

int wubu_failweight_penalty(const wubu_failweight_t *f, int region) {
    if (!f || region < 0 || region >= f->n_regions) return 0;
    return f->regions[region].failures * f->failure_weight - f->regions[region].successes;
}

int wubu_failweight_avoid(const wubu_failweight_t *f, int region) {
    if (!f || region < 0 || region >= f->n_regions) return 1;  /* fail-safe: avoid unknown */
    return wubu_failweight_penalty(f, region) > 0 ? 1 : 0;
}

int wubu_failweight_best(const wubu_failweight_t *f) {
    if (!f || f->n_regions <= 0) return -1;
    int best = 0;
    int best_pen = wubu_failweight_penalty(f, 0);
    for (int r = 1; r < f->n_regions; r++) {
        int pen = wubu_failweight_penalty(f, r);
        if (pen < best_pen) { best_pen = pen; best = r; }
    }
    return best;
}
