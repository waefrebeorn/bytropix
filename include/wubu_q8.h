#ifndef WUBU_Q8_H
#define WUBU_Q8_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Q8_0 block quant: int8 weights + per-32 fp16 scale. Effectively lossless. */
void wubu_q8_quant(const float *x, int8_t *q, uint16_t *scale_f16, int n);
void wubu_q8_dequant(const int8_t *q, const uint16_t *scale_f16, float *x, int n);
float wubu_q8_cosine(const float *a, const float *b, int n);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_Q8_H */
