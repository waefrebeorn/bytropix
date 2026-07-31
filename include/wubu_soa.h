/*
 * wubu_soa.h -- Structure-of-Arrays activation tensor layout (doc I02/C02).
 *
 * Converts activation tensors from AoS to SoA for cache-friendly
 * channel-wise access (gating, scaling, SmoothQuant, AWQ).
 *
 * Self-contained C11, no third-party deps.
 */

#ifndef WUBU_SOA_H
#define WUBU_SOA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Convert AoS [batch][dim] → SoA [dim][batch]. */
void wubu_soa_pack(const float *aos, float *soa, int batch, int dim);

/* Convert SoA [dim][batch] → AoS [batch][dim]. */
void wubu_soa_unpack(const float *soa, float *aos, int batch, int dim);

/* Per-channel scaling in SoA layout (cache-friendly). */
void wubu_soa_scale_channels(float *soa, const float *scale, int batch, int dim);

/* Per-token scaling in SoA layout. */
void wubu_soa_scale_tokens(float *soa, const float *scale, int batch, int dim);

/* Arena-backed SoA allocator (placeholder: uses malloc). */
float *wubu_soa_alloc(void *arena, int batch, int dim);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SOA_H */
