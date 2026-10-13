#ifndef WUBU_NOVELTY_SEARCH_H
#define WUBU_NOVELTY_SEARCH_H

/* wubu_novelty_search.h -- AGENT #164 novelty-search anti-local-optimum.
 *
 * A behavioral novelty-search reward that drives exploration away from local
 * optima: rather than rewarding fitness (which collapses a population into the
 * nearest peak), novelty search rewards *how novel* a behavior is compared to
 * an archive of previously-seen behaviors. Each behavior is a fixed-dim
 * descriptor (e.g. a trajectory projection); novelty is the average distance
 * to the k-nearest neighbors in the archive. The archive holds novel
 * behaviors (frontier expansion), and a minimum-deposit novelty threshold
 * ensures only sufficiently novel behaviors are archived. Deterministic, no
 * stubs. */

#include <stddef.h>

#define WUBU_NS_DIM 8       /* behavior-descriptor dimensionality */
#define WUBU_NS_ARCHIVE_MAX 256
#define WUBU_NS_K 15        /* k-nearest-neighbor count */

typedef struct {
    float v[WUBU_NS_DIM];
} wubu_ns_desc_t;

typedef struct {
    wubu_ns_desc_t desc;
    int fitness;            /* underlying task fitness (auxiliary) */
    int archived;
} wubu_ns_behavior_t;

typedef struct {
    wubu_ns_behavior_t items[WUBU_NS_ARCHIVE_MAX];
    int n;                    /* archive size */
    int k;                    /* current KNN k (<=NS_K) */
    float threshold;          /* min novelty to archive a new behavior */
    float last_novelty;       /* novelty of the most-recent request */
    int n_archived;           /* count of behaviors archived */
    int n_rejected;           /* behaviors below threshold, not archived */
} wubu_novelty_search_t;

/* Initialise: empty archive, k=NS_K, threshold given. */
int wubu_ns_init(wubu_novelty_search_t *ns, float threshold);

/* Add a fixed behavior descriptor to the archive directly. Returns 0/-1. */
int wubu_ns_seed(wubu_novelty_search_t *ns, const float *desc, int fitness);

/* Compute the novelty score of a behavior descriptor = average distance to the
 * k-nearest neighbors already in the archive. Returns 0/-1; score in *out.
 * With an empty archive, novelty is 0 (nothing to be novel *against*). */
int wubu_ns_novelty(const wubu_novelty_search_t *ns,
                    const float *desc, float *out);

/* Evaluate a candidate behavior: novelty must reach `threshold` to be
 * archived; the archive frontier expands only with genuinely novel behaviors
 * (anti-local-optimum). Returns 1 if archived, 0 if rejected (below
 * threshold), -1 on bad args. */
int wubu_ns_evaluate(wubu_novelty_search_t *ns,
                     const float *desc, int fitness);

/* Euclidean distance between two descriptors. */
float wubu_ns_dist(const float *a, const float *b);

/* Archive size / counts. */
int wubu_ns_archive_size(const wubu_novelty_search_t *ns);

#endif
