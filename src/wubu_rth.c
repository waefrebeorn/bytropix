/* wubu_rth.c -- AGIEDR #310 adversarial red-team agent harness.
 * See wubu_rth.h. */
#include "wubu_rth.h"
#include <string.h>
#include <stdio.h>

static uint32_t rng_next(wubu_rth_t *h) {
    h->rng = h->rng * 1103515245U + 12345U;
    return (h->rng >> 16) & 0x7fff;
}

int wubu_rth_init(wubu_rth_t *h, uint32_t seed) {
    if (!h) return -1;
    memset(h->probes, 0, sizeof h->probes);
    h->n = 0;
    h->n_bypasses = 0;
    h->rng = seed ? seed : 1;
    return 0;
}

int wubu_rth_mutate(wubu_rth_t *h, const char *seed, char *out, int cap) {
    if (!h || !seed || !out || cap <= 0) return -1;
    /* pick a mutation strategy */
    int strat = (int)(rng_next(h) % 3);
    const char *obf = "/*x*/";
    const char *pad = " ";
    int len;
    if (strat == 0) {
        /* insert an obfuscation token near the front */
        len = snprintf(out, cap, "%s%s", obf, seed);
    } else if (strat == 1) {
        /* append a trailing token */
        len = snprintf(out, cap, "%s%s", seed, pad);
    } else {
        /* interleave: insert obfuscation in the middle */
        int sl = (int)strlen(seed);
        int mid = sl / 2;
        len = snprintf(out, cap, "%.*s%s%s", mid, seed, obf, seed + mid);
    }
    if (len < 0 || len >= cap) return -1;
    return len;
}

int wubu_rth_probe(wubu_rth_t *h, const char *seed, wubu_rth_defense_t def, void *ctx) {
    if (!h || !seed || !def) return -1;
    if (h->n >= WUBU_RTH_MAX_PROBES) return -1;
    char attack[WUBU_RTH_ATKLEN];
    if (wubu_rth_mutate(h, seed, attack, sizeof attack) < 0) return -1;
    int blocked = def(attack, ctx);
    strncpy(h->probes[h->n].attack, attack, WUBU_RTH_ATKLEN - 1);
    h->probes[h->n].attack[WUBU_RTH_ATKLEN-1] = '\0';
    h->probes[h->n].blocked = blocked ? 1 : 0;
    if (!blocked) h->n_bypasses++;
    h->n++;
    return blocked ? 1 : 0;
}

int wubu_rth_bypasses(const wubu_rth_t *h) {
    return h ? h->n_bypasses : 0;
}

double wubu_rth_bypass_rate(const wubu_rth_t *h) {
    if (!h || h->n == 0) return 0.0;
    return (double)h->n_bypasses / (double)h->n;
}
