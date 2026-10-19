/*
 * wubu_sae.c -- 256-bit signature matching. See wubu_sae.h.
 */
#include "wubu_sae.h"
#include <string.h>

int wubu_sae_match(const uint8_t query[32], const wubu_sae_atom *atoms, int n,
                   int *best_index, int *best_dist)
{
    if (!query || !atoms || !best_index || !best_dist || n <= 0) return -1;

    int bi = -1, bd = 257;
    for (int i = 0; i < n; i++) {
        int d = 0;
        for (int k = 0; k < 32; k++)
            d += __builtin_popcount((unsigned)(query[k] ^ atoms[i].sig[k]));
        if (d < bd) { bd = d; bi = i; if (d == 0) break; }
    }
    *best_index = bi;
    *best_dist = bd;
    return 0;
}

int wubu_sae_match_batch(const uint8_t *queries, int n_queries,
                         const wubu_sae_atom *atoms, int n_atoms,
                         int max_dist, int *out_index, int *out_dist)
{
    if (!queries || !atoms || !out_index || !out_dist) return -1;
    if (n_queries <= 0 || n_atoms <= 0 || max_dist < 0) return -1;

    for (int q = 0; q < n_queries; q++) {
        int bi = -1, bd = 257;
        const uint8_t *sig = queries + (size_t)q * 32;
        for (int i = 0; i < n_atoms; i++) {
            int d = 0;
            for (int k = 0; k < 32; k++)
                d += __builtin_popcount((unsigned)(sig[k] ^ atoms[i].sig[k]));
            if (d < bd) { bd = d; bi = i; }
        }
        if (bd <= max_dist) { out_index[q] = bi; out_dist[q] = bd; }
        else               { out_index[q] = -1; out_dist[q] = bd; }
    }
    return 0;
}

void wubu_sae_signature(const float *v, int n, uint8_t sig[32])
{
    if (!v || !sig || n <= 0) return;
    memset(sig, 0, 32);

    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += v[i];
    float mean = (float)(sum / (double)n);

    int bits = n < 256 ? n : 256;
    for (int i = 0; i < bits; i++)
        if (v[i] > mean) sig[i >> 3] |= (uint8_t)(1u << (i & 7));
}