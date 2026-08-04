/*
 * wubu_hashrouter.h -- hash-based expert routing for MoE (DeepSeek V3.2/V4 style).
 *
 * The 'hash routing' trick: instead of a learned router (which can
 * collapse onto a few experts and needs an auxiliary load-balance loss
 * to stay sane), the expert assignment is a pure function of the token:
 * hash(token_id, position, slot, seed) picks each of the top-k experts.
 * No router weights, no gradients, no aux loss -- balance falls out of
 * the hash being uniform, and the mapping is deterministic for a fixed
 * seed. The top-k slots use distinct per-slot salts so the k chosen
 * experts are always distinct.
 *
 * Self-contained C11, no third-party deps (libc only).
 */
#ifndef WUBU_HASHROUTER_H
#define WUBU_HASHROUTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* opaque hash-router state: n_experts, top_k, seed */
typedef struct wubu_hashrouter wubu_hashrouter_t;

/* build a router. Requires 1 <= top_k <= n_experts; returns NULL on bad
 * args or OOM. seed 0 is legal (folded in as-is). */
wubu_hashrouter_t *wubu_hashrouter_create(int n_experts, int top_k, uint32_t seed);

void wubu_hashrouter_free(wubu_hashrouter_t *hr);

/* route one token: fills out_experts[0..top_k-1] with top_k DISTINCT
 * expert ids in [0, n_experts). Returns top_k on success, -1 on bad
 * args (NULL hr or out_experts). Deterministic for a fixed seed:
 * the same (token_id, pos) always yields the same expert list. */
int wubu_hashrouter_route(const wubu_hashrouter_t *hr, uint32_t token_id,
                          uint32_t pos, int *out_experts);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_HASHROUTER_H */
