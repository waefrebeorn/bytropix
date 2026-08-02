/*
 * wubu_active.c -- Active Learning (uncertainty sampling / QBC) (FF05). C11.
 *
 * Convergence (active learning / query-by-committee 7-hop):
 *   - FF05: uncertainty sampling = query argmax σ(x); QBC = query argmax
 *     committee disagreement. At home: instead of evaluating random configs,
 *     the active learner queries the highest-σ config (most informative),
 *     unifying BO + active learning — same as UCB but framed as label acquisition.
 */
#include "wubu_active.h"
#include <math.h>
#include <string.h>

int wubu_active_uncertainty(const wubu_active_t *al, int *out_idx) {
    if (!al || !out_idx || al->n == 0) return -1;
    double best = -1; int best_i = -1;
    for (int i = 0; i < al->n; i++) {
        if (al->queried[i]) continue;
        if (al->var[i] > best) { best = al->var[i]; best_i = i; }
    }
    if (best_i < 0) return -1;
    *out_idx = best_i;
    return 0;
}

int wubu_active_qbc(const wubu_active_t *al, int *out_idx) {
    if (!al || !out_idx || al->n == 0) return -1;
    int best = -1; int best_i = -1;
    for (int i = 0; i < al->n; i++) {
        if (al->queried[i]) continue;
        if (al->committee_disagree[i] > best) { best = al->committee_disagree[i]; best_i = i; }
    }
    if (best_i < 0) return -1;
    *out_idx = best_i;
    return 0;
}

int wubu_active_query(wubu_active_t *al, int idx) {
    if (!al || idx < 0 || idx >= al->n) return -1;
    al->queried[idx] = 1;
    return 0;
}
