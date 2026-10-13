/* wubu_caprevoke.c -- NAMESPACE #191 capability revocation propagates.
 * See wubu_caprevoke.h. */
#include "wubu_caprevoke.h"
#include <string.h>

int wubu_cr_init(wubu_caprevoke_t *c) {
    if (!c) return -1;
    memset(c->caps, 0, sizeof c->caps);
    c->n = 0;
    c->n_revocations = 0;
    return 0;
}

static wubu_cr_cap_t *find(wubu_caprevoke_t *c, int id) {
    for (int i = 0; i < c->n; i++) if (c->caps[i].id == id) return &c->caps[i];
    return NULL;
}

int wubu_cr_create(wubu_caprevoke_t *c, int parent, int rights) {
    if (!c || c->n >= WUBU_CR_MAX) return -1;
    int effective = rights;
    if (parent >= 0) {
        wubu_cr_cap_t *p = find(c, parent);
        if (!p || p->revoked) return -1;         /* can't derive from revoked/missing */
        effective = rights & p->rights;          /* attenuate to a subset */
    }
    int id = c->n;                               /* dense ids 0..n-1 */
    c->caps[c->n].id = id;
    c->caps[c->n].parent = parent;
    c->caps[c->n].revoked = 0;
    c->caps[c->n].rights = effective;
    c->n++;
    return id;
}

static int is_descendant(const wubu_caprevoke_t *c, int candidate, int ancestor) {
    /* walk up from candidate; if we reach ancestor, it's a descendant */
    int cur = candidate;
    while (cur >= 0) {
        if (cur == ancestor) return 1;
        cur = c->caps[cur].parent;
    }
    return 0;
}

int wubu_cr_revoke(wubu_caprevoke_t *c, int id) {
    if (!c) return -1;
    wubu_cr_cap_t *target = find(c, id);
    if (!target || target->revoked) return -1;
    int revoked = 0;
    for (int i = 0; i < c->n; i++) {
        /* revoke the target and every descendant of it */
        if (c->caps[i].id == id || is_descendant(c, c->caps[i].id, id)) {
            if (!c->caps[i].revoked) {
                c->caps[i].revoked = 1;
                revoked++;
                c->n_revocations++;
            }
        }
    }
    return revoked;
}

int wubu_cr_valid(const wubu_caprevoke_t *c, int id) {
    if (!c) return 0;
    for (int i = 0; i < c->n; i++) if (c->caps[i].id == id) return !c->caps[i].revoked;
    return 0;
}

int wubu_cr_rights(const wubu_caprevoke_t *c, int id) {
    if (!c) return 0;
    for (int i = 0; i < c->n; i++)
        if (c->caps[i].id == id) return c->caps[i].revoked ? 0 : c->caps[i].rights;
    return 0;
}
