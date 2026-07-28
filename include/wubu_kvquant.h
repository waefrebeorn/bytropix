#ifndef WUBU_KVQUANT_H
#define WUBU_KVQUANT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* FP8 e4m3 KV store. */
void wubu_kvquant_fp8_encode(const float *x, int8_t *out, int n, float scale, float *out_scale);
void wubu_kvquant_fp8_decode(const int8_t *in, float *out, int n, float scale);

/* INT4 + Walsh-Hadamard rotation (SAW-INT4 style, near-lossless). */
void wubu_kvquant_int4_encode(const float *x, uint8_t *out, int n, float *out_scale);
void wubu_kvquant_int4_decode(const uint8_t *in, float *out, int n, float scale);

/* Cosine similarity (accuracy check). */
float wubu_kvquant_cosine(const float *a, const float *b, int n);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KVQUANT_H */
