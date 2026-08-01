/*
 * wubu_planediv.c -- Control/data-plane separation + poisoning divergence (AG02/AG03). C11.
 *
 * Convergence (ASI01/LLM01 control/data-plane, ASI06/L3xT3 cross-session 7-hop):
 *   - AG02 control/data-plane separation: every input is tagged control-plane
 *          (instructions the reasoner MAY obey) or data-plane (content it must
 *          NOT treat as instruction). A data-plane item is rejected as an
 *          instruction -> kills goal-hijack / injection.
 *   - AG03 memory poisoning divergence: keep a fingerprint of trusted episodic
 *          memory; a replayed/poisoned memory whose fingerprint diverges from
 *          the trusted baseline is flagged. Cross-session replay detection.
 *
 * Pure C11, deterministic, testable.
 */
#include "wubu_planediv.h"
#include <stdlib.h>
#include <string.h>

/* AG02: classify + enforce plane. Returns
 *   1 = instruction accepted (control-plane),
 *   0 = rejected as instruction (data-plane => cannot drive actions). */
int wubu_plane_enforce(const wubu_plane_t *p, int item_plane, const char *content) {
    (void)content;
    if (!p) return 0;
    if (item_plane == WUBU_PLANE_CONTROL) return 1;   /* trusted instruction */
    /* data-plane: only obey if explicitly permitted by policy (default deny) */
    return p->allow_data_as_instruction ? 1 : 0;
}

/* AG03: fingerprint a memory blob (FNV-1a 64). */
unsigned long long wubu_mem_fingerprint(const char *blob, int n) {
    unsigned long long h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) { h ^= (unsigned char)blob[i]; h *= 1099511628211ULL; }
    return h;
}

/* AG03: diverged? trustworthy baseline fp vs current; if mismatch -> poisoned. */
int wubu_mem_diverged(unsigned long long trusted_fp, unsigned long long cur_fp) {
    return (trusted_fp != cur_fp) ? 1 : 0;
}

/* AG03: cross-session replay flag: same content fingerprint appearing in a new
 * session context is flagged for review (returns 1 if flagged). */
int wubu_replay_flagged(unsigned long long fp, const unsigned long long *seen,
                        int n_seen) {
    for (int i = 0; i < n_seen; i++) if (seen[i] == fp) return 1;
    return 0;
}
