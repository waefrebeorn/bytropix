/*
 * wubu_soa.c -- Structure-of-Arrays activation tensor layout (doc I02/C02).
 *
 * Converts activation tensors from AoS (Array of Structs) to SoA (Struct of
 * Arrays) for cache-friendly channel-wise access. In AoS, token i's hidden
 * vector is contiguous [d0, d1, ..., dN]. In SoA, channel j's values across
 * all tokens is contiguous [t0_j, t1_j, ..., tM_j].
 *
 * Win: channel-wise operations (gating, scaling, smoothquant, AWQ) get
 * sequential memory access instead of strided, improving cache hit rate.
 *
 * Self-contained C11, no third-party deps.
 */

#include "wubu_soa.h"
#include <stdlib.h>
#include <string.h>

/* Convert a batch of activations from AoS to SoA layout.
 * Input:  aos[batch * dim] — each row is one token's hidden vector
 * Output: soa[dim * batch] — each row is one channel's values across all tokens
 */
void wubu_soa_pack(const float *aos, float *soa, int batch, int dim) {
    if (!aos || !soa || batch <= 0 || dim <= 0) return;
    for (int c = 0; c < dim; c++) {
        for (int t = 0; t < batch; t++) {
            soa[c * batch + t] = aos[t * dim + c];
        }
    }
}

/* Convert SoA back to AoS (for interfacing with existing engine paths). */
void wubu_soa_unpack(const float *soa, float *aos, int batch, int dim) {
    if (!soa || !aos || batch <= 0 || dim <= 0) return;
    for (int t = 0; t < batch; t++) {
        for (int c = 0; c < dim; c++) {
            aos[t * dim + c] = soa[c * batch + t];
        }
    }
}

/* Apply per-channel scaling in SoA layout (cache-friendly: each channel
 * is contiguous, so scale[c] applies to a contiguous run of values).
 * soa[c * batch + t] *= scale[c] for all t. */
void wubu_soa_scale_channels(float *soa, const float *scale, int batch, int dim) {
    if (!soa || !scale || batch <= 0 || dim <= 0) return;
    for (int c = 0; c < dim; c++) {
        float s = scale[c];
        float *row = soa + c * batch;
        for (int t = 0; t < batch; t++) {
            row[t] *= s;
        }
    }
}

/* Apply per-token scaling in SoA layout.
 * soa[c * batch + t] *= scale[t] for all c. */
void wubu_soa_scale_tokens(float *soa, const float *scale, int batch, int dim) {
    if (!soa || !scale || batch <= 0 || dim <= 0) return;
    for (int c = 0; c < dim; c++) {
        float *row = soa + c * batch;
        for (int t = 0; t < batch; t++) {
            row[t] *= scale[t];
        }
    }
}

/* Arena-backed SoA allocator: allocates a contiguous SoA block from an arena.
 * Returns pointer to the SoA buffer (dim * batch floats). */
float *wubu_soa_alloc(void *arena, int batch, int dim) {
    (void)arena;  /* real impl would use wubu_arena_malloc */
    return (float *)malloc(sizeof(float) * batch * dim);
}
