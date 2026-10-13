/* wubu_novelty_search.c -- AGENT #164 novelty-search anti-local-optimum.
 * See wubu_novelty_search.h. Real, deterministic behavioral novelty search. */

#include "wubu_novelty_search.h"
#include <math.h>
#include <string.h>

float wubu_ns_dist(const float *a, const float *b) {
    float s = 0.0f;
    for (int i = 0; i < WUBU_NS_DIM; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return sqrtf(s);
}

int wubu_ns_init(wubu_novelty_search_t *ns, float threshold) {
    if (!ns) return -1;
    memset(ns, 0, sizeof(*ns));
    ns->k = WUBU_NS_K;
    ns->threshold = threshold;
    return 0;
}

int wubu_ns_seed(wubu_novelty_search_t *ns, const float *desc, int fitness) {
    if (!ns || !desc) return -1;
    if (ns->n >= WUBU_NS_ARCHIVE_MAX) return -1;
    wubu_ns_behavior_t *b = &ns->items[ns->n++];
    memset(b, 0, sizeof(*b));
    memcpy(b->desc.v, desc, sizeof(b->desc.v));
    b->fitness = fitness;
    b->archived = 1;
    ns->n_archived++;
    return 0;
}

int wubu_ns_novelty(const wubu_novelty_search_t *ns,
                    const float *desc, float *out) {
    if (!ns || !desc || !out) return -1;
    if (ns->n == 0) { *out = 0.0f; return 0; }   /* nothing to be novel against */
    int k = ns->k;
    if (k > ns->n) k = ns->n;
    /* collect distances to all archived behaviors */
    float dists[WUBU_NS_ARCHIVE_MAX];
    for (int i = 0; i < ns->n; i++)
        dists[i] = wubu_ns_dist(desc, ns->items[i].desc.v);
    /* exact duplicate of an archived behavior -> novelty 0 (already seen) */
    for (int i = 0; i < ns->n; i++)
        if (dists[i] == 0.0f) { *out = 0.0f; return 0; }
    /* k-nearest: selection of the k smallest via partial sort */
    float sum = 0.0f;
    for (int j = 0; j < k; j++) {
        int best = j;
        for (int i = j + 1; i < ns->n; i++)
            if (dists[i] < dists[best]) best = i;
        float tmp = dists[j]; dists[j] = dists[best]; dists[best] = tmp;
        sum += dists[j];
    }
    *out = sum / (float)k;
    return 0;
}

int wubu_ns_evaluate(wubu_novelty_search_t *ns,
                     const float *desc, int fitness) {
    if (!ns || !desc) return -1;
    float nov;
    if (wubu_ns_novelty(ns, desc, &nov) != 0) return -1;
    ns->last_novelty = nov;
    if (nov >= ns->threshold) {
        /* novel enough: archive it (frontier expansion), fail-closed if full */
        if (ns->n < WUBU_NS_ARCHIVE_MAX) {
            wubu_ns_behavior_t *b = &ns->items[ns->n++];
            memset(b, 0, sizeof(*b));
            memcpy(b->desc.v, desc, sizeof(b->desc.v));
            b->fitness = fitness;
            b->archived = 1;
            ns->n_archived++;
            return 1;   /* archived */
        }
        /* archive full: cannot expand frontier -> reject (fail-closed) */
        ns->n_rejected++;
        return 0;
    }
    ns->n_rejected++;
    return 0;       /* below threshold -> rejected (not novel) */
}

int wubu_ns_archive_size(const wubu_novelty_search_t *ns) {
    return ns ? ns->n : -1;
}
