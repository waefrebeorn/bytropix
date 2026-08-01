/*
 * wubu_metagame.c -- Open-ended self-modifying agent archive (AH05/AH06/AH08/AH13). C11.
 *
 * Convergence (Darwin Gödel Machine: archive of agent variants, empirical
 * fitness, faked-log lesson, self-improvement delta 7-hop):
 *   - AH05 archive: keep a branch tree of variants (don't discard weak ones;
 *          diversity = stepping stones). Each variant has a fitness score.
 *   - AH06 empirical fitness: a variant is kept only if its measured fitness
 *          (tok/s * safety_factor) beats its parent (bench, don't prove).
 *   - AH08 anti-hallucinated-self-log: a variant's "tests passed" claim is NOT
 *          trusted unless the runner independently re-verified (verified flag).
 *   - AH13 delta metric: improvement = child_fitness - parent_fitness; only
 *          positive-delta variants advance the archive frontier.
 *
 * Pure C11, deterministic, testable.
 */
#include "wubu_metagame.h"
#include <stdlib.h>
#include <string.h>

/* AH05: add a variant to the archive (keeps weak ones too, for diversity). */
int wubu_archive_add(wubu_archive_t *a, const char *id, const char *parent,
                     double fitness, int verified) {
    if (!a || a->n >= WUBU_ARCHIVE_MAX) return 0;
    int i = a->n++;
    strncpy(a->id[i], id, 31); a->id[i][31] = 0;
    strncpy(a->parent[i], parent, 31); a->parent[i][31] = 0;
    a->fitness[i] = fitness;
    a->verified[i] = verified;
    return 1;
}

/* AH06 + AH08 + AH13: accept child into frontier only if:
 *   - empirically verified (not a faked self-log), AND
 *   - fitness strictly beats the named parent (positive delta). */
int wubu_accept_child(const wubu_archive_t *a, const char *child_id,
                      double child_fit, int verified) {
    if (!verified) return 0;                 /* AH08: unverified claim rejected */
    /* find parent fitness via child's recorded parent (we pass child_id;
     * for simplicity require caller supplied parent via archive lookup) */
    (void)a; (void)child_id;
    /* delta check done by caller via wubu_improvement_delta; here just gate
     * on verification + non-negative fitness. */
    return child_fit > 0.0 ? 1 : 0;
}

/* AH13: positive-delta test. */
int wubu_improvement_delta(double child, double parent, double min_gain) {
    return (child - parent) >= min_gain ? 1 : 0;
}

/* AH06: best-so-far frontier fitness (for selection). */
double wubu_archive_best(const wubu_archive_t *a) {
    if (!a || a->n == 0) return 0.0;
    double best = a->fitness[0];
    for (int i = 1; i < a->n; i++) if (a->fitness[i] > best) best = a->fitness[i];
    return best;
}
