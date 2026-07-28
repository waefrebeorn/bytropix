/*
 * wubu_gemv_tune.h -- Roofline-driven GEMV auto-tuner.
 *
 * Convergent basis (Kevin-Bacon meta-analysis, matmul half):
 *   - Roofline (2607.02558): decode GEMV is memory-bandwidth-bound; the
 *     lever is WEIGHT PRECISION (halve weight bytes -> halve the dominant
 *     traffic) and KERNEL TILE (match the SIMD width / cache line).
 *   - llama.cpp / Intel Xeon: int8 weight GEMV ~2x decode throughput when
 *     BW-bound; AVX512 (16-wide) beats AVX2 (8-wide) on the K-reduction.
 *
 * So: pick k_unroll from the CPU SIMD width, and pick int8 when the B*-ridge
 * says the matmul is bandwidth-bound for this (M,K) on this hardware.
 */
#ifndef WUBU_GEMV_TUNE_H
#define WUBU_GEMV_TUNE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int k_unroll;     /* SIMD K-reduction unroll (8 AVX2, 16 AVX512) */
    int use_int8;     /* 1 = quantize weights to int8 (halve traffic) */
    int avx512;       /* cpu detected AVX512F */
} wubu_gemv_tile_t;

/* Detect SIMD width + AVX512 from the CPU. */
wubu_gemv_tile_t wubu_gemv_detect(void);

/* Auto-tune for a GEMV of shape (M output rows, K reduce).
 * beta_eff_tb_s = effective memory bandwidth in TB/s (0 = auto-detect guess).
 * Returns the tile to use. use_int8 is set when BW-bound enough that the
 * int8 halving wins over the requant cost for this shape. */
wubu_gemv_tile_t wubu_gemv_autotune(int M, int K, double beta_eff_tb_s);

/* Human-readable tile description (static buffer). */
const char *wubu_gemv_tile_name(const wubu_gemv_tile_t *t);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_GEMV_TUNE_H */
