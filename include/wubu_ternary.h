/*
 * wubu_ternary.h -- BitNet 1.58 ternary {-1,0,+1} weight GEMV (doc 004).
 *
 * WHY (Kevin-Bacon convergence): ternary weights are the floor of the weight
 * bytes/token curve -- 2 bits/weight (stored as 2-bit packed, or 1.58 avg with
 * the zero-skip). BitNet 1.58 (Ma et al., 2023) shows near-lossless 1.58-bit
 * LLM inference: W is rounded to {-1,0,+1} with a per-block absmean scale, then
 * the matmul is W_ternary @ x * scale. This is the complement to B02 (int4) on
 * the WEIGHT side: even smaller, and the matmul becomes mostly adds (no full
 * multiplies for the +/-1 entries).
 *
 * SCHEME (own-C, data-independent scaling): per output-row (or per block)
 *   scale = mean(|W_row|);  w_q = clip(round(W_row / scale), -1, 1) in {-1,0,+1}.
 * GEMV: y = scale * (W_q @ x).  Stored packed 2 bits/weight.
 *
 * Importantly this is SAFE on CPU (no HW requirement) and the oracle is exact:
 * ternary W_q @ x * scale == reference fp32 within float rounding for the
 * quantized weights (cosine to the ORIGINAL fp32 W @ x depends on quantization
 * error, bounded by the absmean rounding -- typically cosine > 0.97 on real
 * linear layers).
 */
#ifndef WUBU_TERNARY_H
#define WUBU_TERNARY_H

#include <stdint.h>
#include <stddef.h>

/* Packed ternary weights for an [M, K] matrix: 2 bits/weight, 4 weights/byte.
 * value 0,1,2,3 -> {-1,0,+1,pad}. The 4th slot (value 3) is an unused pad so
 * padding to multiples of 4 is free. */
typedef struct {
    int M, K;
    int K_packed;          /* ceil(K/4) */
    int8_t  *t;            /* [M * K_packed] packed 2-bit weights */
    float   *scale;        /* [M] per-row absmean scale */
} wubu_ternary_t;

/* Quantize an [M,K] fp32 matrix to ternary. Returns 0 on success. */
int wubu_ternary_quantize(const float *W, int M, int K, wubu_ternary_t *q);

/* GEMV: y[M] = scale * (W_q @ x). x has K elements. */
void wubu_ternary_gemv(const wubu_ternary_t *q, const float *x, float *y);

/* Packed-buffer helpers (2 bits/weight, 4 per byte, MSB-first within nibble). */
int  wubu_ternary_packed_bytes(int K);
/* en/decode one row of K weights into a packed buffer (K_packed bytes). */
void wubu_ternary_pack_row(const int8_t *wq, int K, uint8_t *out);
void wubu_ternary_unpack_row(const uint8_t *buf, int K, int8_t *wq_out);

void wubu_ternary_free(wubu_ternary_t *q);

#endif /* WUBU_TERNARY_H */
