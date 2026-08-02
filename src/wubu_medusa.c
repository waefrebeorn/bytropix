/*
 * wubu_medusa.c -- Medusa self-draft heads (parallel tree draft) (HH05). C11.
 *
 * Convergence (Medusa / self-speculative decoding 7-hop):
 *   - HH05: attach lightweight draft heads to the target's last layer → propose
 *     K tokens in PARALLEL (no separate draft model). Tree attention verifies
 *     the draft tree. Adaptive draft length via acceptance-history EMA. At home:
 *     instead of a small draft model, attach draft heads to the 27-layer model
 *     → self-speculation, no extra model, faster verification → higher tok/s.
 */
#include "wubu_medusa.h"
#include <string.h>

int wubu_medusa_init(wubu_medusa_t *m) {
    if (!m) return -1;
    memset(m, 0, sizeof(*m));
    m->n_heads = WUBU_MEDUSA_HEADS;
    m->branch = WUBU_MEDUSA_BRANCH;
    for (int h = 0; h < m->n_heads; h++) m->accept_ema[h] = 0.8f;  /* optimistic */
    m->draft_len = m->n_heads;
    return 0;
}

int wubu_medusa_adapt(wubu_medusa_t *m, float threshold) {
    if (!m) return -1;
    int eff = 0;
    for (int h = 0; h < m->n_heads; h++)
        if (m->accept_ema[h] >= threshold) eff++;
    m->draft_len = (eff == 0) ? 1 : eff;  /* always at least 1 */
    return m->draft_len;
}

int wubu_medusa_update(wubu_medusa_t *m, int head, int accepted, int proposed) {
    if (!m || head < 0 || head >= m->n_heads || proposed <= 0) return -1;
    float rate = (float)accepted / (float)proposed;
    /* EMA: α = 0.1 */
    m->accept_ema[head] = 0.9f * m->accept_ema[head] + 0.1f * rate;
    return 0;
}
