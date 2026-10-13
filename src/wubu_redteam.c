/* wubu_redteam.c -- AGIEDR #303 red-team benchmark gate.
 * See wubu_redteam.h. */
#include "wubu_redteam.h"
#include <string.h>

int wubu_rt_init(wubu_redteam_t *r, double threshold) {
    if (!r) return -1;
    memset(r->probes, 0, sizeof r->probes);
    r->n = 0;
    r->threshold = (threshold >= 0.0 && threshold <= 1.0) ? threshold : 0.9;
    return 0;
}

int wubu_rt_add(wubu_redteam_t *r, const char *payload, int malicious) {
    if (!r || !payload || r->n >= WUBU_RT_MAX_PROBES) return -1;
    r->probes[r->n].payload = payload;
    r->probes[r->n].malicious = malicious ? 1 : 0;
    r->n++;
    return 0;
}

double wubu_rt_run(wubu_redteam_t *r, int *out_correct, int *out_total) {
    if (!r) return 0.0;
    int correct = 0;
    for (int i = 0; i < r->n; i++) {
        int blocked = wubu_indirect_pi_block(r->probes[i].payload);
        /* correct if: malicious probe was blocked, OR benign probe was not blocked */
        if ((r->probes[i].malicious && blocked) || (!r->probes[i].malicious && !blocked))
            correct++;
    }
    if (out_correct) *out_correct = correct;
    if (out_total) *out_total = r->n;
    return r->n > 0 ? (double)correct / (double)r->n : 0.0;
}

int wubu_rt_pass(const wubu_redteam_t *r, double accuracy) {
    if (!r) return 0;   /* fail-closed */
    return accuracy >= r->threshold ? 1 : 0;
}
