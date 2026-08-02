/*
 * wubu_quantkv.h -- KV cache quantization (INT8 group-wise) (HH06).
 */
#ifndef WUBU_QUANTKV_H
#define WUBU_QUANTKV_H

#define WUBU_QUANTKV_GROUP 32   /* per-head group size for scales */
#define WUBU_QUANTKV_MAX 262144 /* max KV elements (512K ctx × heads × 2) */

typedef struct {
    int n;                  /* number of FP32 elements */
    int group;              /* group size for scales */
    signed char q[WUBU_QUANTKV_MAX];   /* INT8 quantized */
    float scale[WUBU_QUANTKV_MAX / WUBU_QUANTKV_GROUP + 1];
    float zero;             /* zero point (symmetric = 0) */
} wubu_quantkv_t;

/* Quantize FP32 KV → INT8 group-wise (symmetric). Returns 0 ok. */
int  wubu_quantkv_quantize(wubu_quantkv_t *qk, const float *kv, int n);
/* Dequantize back to FP32 (for attention compute). */
int  wubu_quantkv_dequantize(const wubu_quantkv_t *qk, float *out);
/* Bits per element (8 for INT8). */
int  wubu_quantkv_bits(void);
/* Compression ratio vs FP32 (32 / 8 = 4x). */
float wubu_quantkv_ratio(void);

#endif