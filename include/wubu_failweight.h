#ifndef WUBU_FAILWEIGHT_H
#define WUBU_FAILWEIGHT_H

/* wubu_failweight.h -- AGENT #148 DA-2 failure weighting 3x bounds search.
 *
 * Failure-weighted search: candidate mutations that FAILED are weighted 3x
 * more heavily when computing the search's avoidance signal, so the search
 * BOUNDS AWAY from regions that keep failing (instead of resampling them).
 * The module tracks per-region success/failure counts and produces a weighted
 * penalty per region = failures * 3 (failures dominate successes 3:1). A
 * candidate in a high-failure region gets a high penalty -> the search avoids
 * it. Deterministic, no allocation, real work (no stubs). */

#define WUBU_FAILWEIGHT_MAX_REGIONS 256

typedef struct {
    int    successes;
    int    failures;
} wubu_region_stat_t;

typedef struct {
    wubu_region_stat_t regions[WUBU_FAILWEIGHT_MAX_REGIONS];
    int  n_regions;
    int  failure_weight;   /* the 3x multiplier (default 3) */
} wubu_failweight_t;

/* Initialise with a region count and failure weight (use 3 for 3x). */
int wubu_failweight_init(wubu_failweight_t *f, int n_regions, int failure_weight);

/* Record a mutation outcome in a region. */
int wubu_failweight_record(wubu_failweight_t *f, int region, int success);

/* Weighted penalty for a region (failures*weight - successes). Higher = avoid.
 * Returns 0 for an unknown/empty region (fail-safe). */
int wubu_failweight_penalty(const wubu_failweight_t *f, int region);

/* Should the search AVOID this region? (penalty > 0 means failures dominate). */
int wubu_failweight_avoid(const wubu_failweight_t *f, int region);

/* Best region to sample next (lowest penalty, ties -> lowest index), or -1. */
int wubu_failweight_best(const wubu_failweight_t *f);

#endif
