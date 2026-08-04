/*
 * wubu_hashrouter.c -- hash-based expert routing (DeepSeek V3.2/V4 style).
 *
 * No learned router: the expert assignment is a pure content hash of the
 * token. Slot k hashes (token_id, pos, salt_k, seed) with our own
 * splitmix64 fold; the k slots use distinct salts, and a collision is
 * re-hashed with a bumped salt, so the top-k experts are always distinct.
 * Deterministic for a fixed seed, learner-free, no aux loss.
 */
#include "wubu_hashrouter.h"

#include <stdlib.h>

struct wubu_hashrouter {
    int n_experts;
    int top_k;
    uint32_t seed;
};

/* splitmix64: stateful 64-bit mixer (public-domain algorithm, written
 * out by hand here). Each call advances the state by the golden-ratio
 * constant, then applies three bijective scrambles. */
static uint64_t wubu_splitmix64(uint64_t *state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* hash (token_id, pos, salt, seed) -> 64-bit. All four inputs are folded
 * into the splitmix state, then two full rounds avalanche the result so
 * a 1-bit change anywhere in the inputs flips ~half the output bits. */
static uint64_t wubu_hashroute_slot(uint32_t token_id, uint32_t pos,
                                    uint32_t salt, uint32_t seed)
{
    uint64_t state = ((uint64_t)seed << 32) ^ (uint64_t)token_id;
    state ^= (uint64_t)pos * 0x9E3779B97F4A7C15ull;
    state ^= (uint64_t)salt * 0xBF58476D1CE4E5B9ull;
    wubu_splitmix64(&state);          /* round 1: avalanche */
    return wubu_splitmix64(&state);   /* round 2: decouple */
}

wubu_hashrouter_t *wubu_hashrouter_create(int n_experts, int top_k, uint32_t seed)
{
    if (n_experts < 1 || top_k < 1 || top_k > n_experts) return NULL;
    wubu_hashrouter_t *hr = (wubu_hashrouter_t *)malloc(sizeof(*hr));
    if (!hr) return NULL;
    hr->n_experts = n_experts;
    hr->top_k = top_k;
    hr->seed = seed;
    return hr;
}

void wubu_hashrouter_free(wubu_hashrouter_t *hr)
{
    free(hr);
}

int wubu_hashrouter_route(const wubu_hashrouter_t *hr, uint32_t token_id,
                          uint32_t pos, int *out_experts)
{
    if (!hr || !out_experts) return -1;

    for (int k = 0; k < hr->top_k; k++) {
        uint32_t salt = (uint32_t)k;   /* per-slot salt */
        for (;;) {
            int e = (int)(wubu_hashroute_slot(token_id, pos, salt, hr->seed)
                          % (uint64_t)hr->n_experts);
            /* already chosen by an earlier slot? re-hash with a new salt */
            int taken = 0;
            for (int j = 0; j < k; j++) {
                if (out_experts[j] == e) { taken = 1; break; }
            }
            if (!taken) { out_experts[k] = e; break; }
            salt += (uint32_t)hr->top_k;
        }
    }
    return hr->top_k;
}
