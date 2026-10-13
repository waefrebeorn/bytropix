/* wubu_humangate.c -- AGENT #146 DA-3 human gate + freeze (death condition).
 * See wubu_humangate.h. */
#include "wubu_humangate.h"

int wubu_humangate_init(wubu_humangate_t *g, uint64_t freeze_token) {
    if (!g) return -1;
    g->frozen = 0;
    g->veto_count = 0;
    g->freeze_token = freeze_token;
    return 0;
}

int wubu_humangate_veto(wubu_humangate_t *g) {
    if (!g) return -1;
    g->frozen = 1;          /* kernel-owned, one-way to set */
    g->veto_count++;
    return 0;
}

int wubu_humangate_may_mutate(const wubu_humangate_t *g) {
    if (!g) return 0;       /* fail-closed: unknown gate denies */
    return g->frozen ? 0 : 1;
}

int wubu_humangate_unfreeze(wubu_humangate_t *g, uint64_t token) {
    if (!g) return 0;
    if (token != g->freeze_token) return 0;   /* wrong token: stays frozen */
    g->frozen = 0;
    return 1;
}

int wubu_humangate_frozen(const wubu_humangate_t *g) {
    return g ? g->frozen : 1;   /* fail-closed: unknown = frozen */
}

int wubu_humangate_vetoes(const wubu_humangate_t *g) {
    return g ? g->veto_count : 0;
}
