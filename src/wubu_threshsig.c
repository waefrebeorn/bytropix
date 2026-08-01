/*
 * wubu_threshsig.c -- Threshold signing (aggregate agent signatures) (DD02). C11.
 *
 * Convergence (threshold cryptography + BLS aggregation 7-hop):
 *   - DD02: simplified threshold signature scheme. Each agent produces a
 *     deterministic pseudo-signature (signer_id ⊕ message_hash) — this is
 *     NOT cryptographically secure (no real BLS/ECDSA), but models the
 *     aggregation mechanism: collect ≥ threshold unique signatures, then
 *     the aggregate is "verified". In production this would be replaced
 *     with actual BLS-137 signatures, but the C11 aggregation logic is
 *     identical (append unique signer, check count ≥ threshold).
 */
#include "wubu_threshsig.h"
#include "wubu_bft.h"
#include <string.h>

int wubu_threshsig_init(wubu_threshsig_t *ts, int n_nodes) {
    if (!ts || n_nodes < 1 || n_nodes > WUBU_THRESHSIG_MAX_SIGS) return -1;
    memset(ts, 0, sizeof(*ts));
    ts->threshold = wubu_bft_threshold(n_nodes);
    ts->n_sigs = 0;
    return 0;
}

unsigned wubu_threshsig_sign(int signer_id, unsigned message_hash) {
    /* Simplified pseudo-signature: XOR-based, deterministic. Not secure. */
    return (((unsigned)signer_id * 2654435761U) ^ message_hash) & 0x7fffffff;
}

int wubu_threshsig_add(wubu_threshsig_t *ts, int signer_id, unsigned message_hash) {
    if (!ts || signer_id < 0) return -1;
    if (ts->n_sigs >= WUBU_THRESHSIG_MAX_SIGS) return -1;
    /* Check not already signed */
    for (int i = 0; i < ts->n_sigs; i++)
        if (ts->sigs[i].signer_id == signer_id) return -1;
    ts->sigs[ts->n_sigs].signer_id = signer_id;
    ts->sigs[ts->n_sigs].sig = wubu_threshsig_sign(signer_id, message_hash);
    ts->n_sigs++;
    return 0;
}

int wubu_threshsig_verified(const wubu_threshsig_t *ts) {
    if (!ts) return 0;
    return (ts->n_sigs >= ts->threshold) ? 1 : 0;
}
