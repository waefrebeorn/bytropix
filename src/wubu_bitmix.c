/* wubu_bitmix.c -- BRAIN #80 mixed per-role bit ladders, never uniform.
 * See wubu_bitmix.h. */
#include "wubu_bitmix.h"

static int clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

int wubu_bitmix_plan(wubu_bitmix_t *m, int base_width, int budget_bits) {
    if (!m || base_width < 2 || base_width > 16) return -1;
    if (budget_bits < WUBU_ROLE_COUNT * 2) return -1;   /* need >=2 bits/role */

    /* sensitivity tier per role: higher = more bits needed.
     * norms/biases full precision; embed/attn sensitive; FFN/MoE robust. */
    int tier[WUBU_ROLE_COUNT] = {
        [WUBU_ROLE_EMBED]     = 2,   /* sensitive */
        [WUBU_ROLE_ATTN_QKV]  = 2,   /* sensitive */
        [WUBU_ROLE_ATTN_OUT]  = 1,
        [WUBU_ROLE_FFN_GATE]  = 0,   /* robust */
        [WUBU_ROLE_FFN_DOWN]  = 0,   /* robust */
        [WUBU_ROLE_MOE_EXPERT]= -1,  /* most robust */
        [WUBU_ROLE_NORM]      = 3,   /* full precision */
        [WUBU_ROLE_BIAS]      = 3,   /* full precision */
    };
    /* width = base + tier offset, norms/biases pinned high (16) */
    for (int r = 0; r < WUBU_ROLE_COUNT; r++) {
        int w;
        if (r == WUBU_ROLE_NORM || r == WUBU_ROLE_BIAS) w = 16;
        else w = base_width + tier[r];
        m->widths[r] = clamp(w, 2, 16);
    }
    /* enforce the budget: if over, shave the most-robust roles first */
    int total = wubu_bitmix_total(m);
    int order[WUBU_ROLE_COUNT] = {WUBU_ROLE_MOE_EXPERT, WUBU_ROLE_FFN_DOWN,
        WUBU_ROLE_FFN_GATE, WUBU_ROLE_ATTN_OUT, WUBU_ROLE_ATTN_QKV,
        WUBU_ROLE_EMBED, WUBU_ROLE_BIAS, WUBU_ROLE_NORM};
    for (int i = 0; i < WUBU_ROLE_COUNT && total > budget_bits; i++) {
        int r = order[i];
        while (m->widths[r] > 2 && total > budget_bits) { m->widths[r]--; total--; }
    }
    m->budget_bits = budget_bits;

    /* guarantee non-uniform: if all widths collapsed to one value, bump the
     * most sensitive role up by 1 (within budget if possible). */
    if (wubu_bitmix_distinct(m) < 2) {
        m->widths[WUBU_ROLE_EMBED] = clamp(m->widths[WUBU_ROLE_EMBED] + 1, 2, 16);
    }
    return 0;
}

int wubu_bitmix_width(const wubu_bitmix_t *m, wubu_role_t role) {
    if (!m || role < 0 || role >= WUBU_ROLE_COUNT) return -1;
    return m->widths[role];
}

int wubu_bitmix_distinct(const wubu_bitmix_t *m) {
    if (!m) return 0;
    int seen[17] = {0};
    int distinct = 0;
    for (int r = 0; r < WUBU_ROLE_COUNT; r++) {
        int w = m->widths[r];
        if (w >= 2 && w <= 16 && !seen[w]) { seen[w] = 1; distinct++; }
    }
    return distinct;
}

int wubu_bitmix_total(const wubu_bitmix_t *m) {
    if (!m) return 0;
    int total = 0;
    for (int r = 0; r < WUBU_ROLE_COUNT; r++) total += m->widths[r];
    return total;
}
