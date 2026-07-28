/*
 * wubu_kv_select.h -- Roofline-driven KV-cache + weight compression selector.
 *
 * Convergent basis (Kevin-Bacon meta-analysis):
 *   - Roofline (2607.02558): decode is BW-bound; the B*-crossover says
 *     compress weights when batch is small, compress KV when context is long.
 *   - DB buffer-pool + llama.cpp + KIVI: KV is the hot buffer; quantize it.
 *     K and V need DIFFERENT schemes (KIVI: V per-token is most sensitive).
 *
 * So given model params P, batch B, context s, bandwidth beta:
 *   - if roofline says COMPRESS_KV  -> pick KV_CACHE_KIVI (V per-token) or
 *                                     KV_CACHE_OUR_Q8 (near-lossless) depending
 *                                     on how BW-bound we are (KIVI when very long).
 *   - if COMPRESS_WEIGHTS           -> recommend int4 weights (handled elsewhere)
 *   - else                          -> KV_CACHE_F16 (ample bandwidth).
 */
#ifndef WUBU_KV_SELECT_H
#define WUBU_KV_SELECT_H

#include "wubu_roofline.h"

typedef enum {
    WUBU_KV_F16 = 0,    /* ample bandwidth: keep fp16 KV */
    WUBU_KV_Q8  = 1,    /* our Q8_0 block-32: near-lossless, 2x vs fp16 */
    WUBU_KV_KIVI = 2     /* KIVI K!=V: best for very long / very BW-bound */
} wubu_kv_scheme_t;

typedef struct {
    wubu_kv_scheme_t kv;
    int              kv_bits;     /* effective KV bits for roofline re-entry */
    int              weight_bits; /* 16 (fp16) or 4 (int4) if compress weights */
    const char      *why;         /* human-readable reason (for logging) */
} wubu_kv_choice_t;

/* Choose a KV scheme + weight precision from the roofline crossover.
 * cfg: hardware/model config. P_params: model params (billions). B: batch.
 * s: context length (tokens). */
wubu_kv_choice_t wubu_kv_select(const wubu_roofline_cfg_t *cfg,
                                 double P_params, int B, int s);

/* String name for a scheme (logging). */
const char *wubu_kv_scheme_name(wubu_kv_scheme_t kv);

#endif /* WUBU_KV_SELECT_H */
