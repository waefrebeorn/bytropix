/* wubu_ztv.c -- NAMESPACE #193 zero-trust continuous verification.
 * See wubu_ztv.h. */
#include "wubu_ztv.h"
#include <string.h>

int wubu_ztv_init(wubu_ztv_t *z) {
    if (!z) return -1;
    memset(z->caps, 0, sizeof z->caps);
    z->n = 0;
    z->current_epoch = 0;
    z->n_allowed = 0;
    z->n_denied = 0;
    return 0;
}

int wubu_ztv_issue(wubu_ztv_t *z, int cap_id) {
    if (!z || z->n >= WUBU_ZT_MAX_CAPS) return -1;
    for (int i = 0; i < z->n; i++) if (z->caps[i].cap_id == cap_id) return -1;
    z->caps[z->n].cap_id = cap_id;
    z->caps[z->n].revoked = 0;
    z->caps[z->n].epoch = z->current_epoch;
    z->n++;
    return 0;
}

int wubu_ztv_revoke(wubu_ztv_t *z, int cap_id) {
    if (!z) return -1;
    for (int i = 0; i < z->n; i++) {
        if (z->caps[i].cap_id == cap_id) {
            z->caps[i].revoked = 1;
            z->current_epoch++;   /* bump epoch: invalidates freshness of ALL */
            return 0;
        }
    }
    return -1;   /* unknown cap */
}

int wubu_ztv_verify(wubu_ztv_t *z, int cap_id) {
    if (!z) return 0;   /* fail-closed */
    for (int i = 0; i < z->n; i++) {
        if (z->caps[i].cap_id == cap_id) {
            /* re-verify: not revoked AND fresh (epoch matches current) */
            if (z->caps[i].revoked) { z->n_denied++; return 0; }
            if (z->caps[i].epoch != z->current_epoch) { z->n_denied++; return 0; }
            z->n_allowed++;
            return 1;
        }
    }
    z->n_denied++;   /* unknown cap */
    return 0;
}

void wubu_ztv_stats(const wubu_ztv_t *z, int *allowed, int *denied) {
    if (!z) return;
    if (allowed) *allowed = z->n_allowed;
    if (denied) *denied = z->n_denied;
}
