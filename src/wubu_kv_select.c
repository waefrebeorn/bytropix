/*
 * wubu_kv_select.c -- Roofline-driven KV/weight selector (see header).
 * Pure C, routes through the tested wubu_roofline module.
 */
#include "wubu_kv_select.h"
#include <string.h>

const char *wubu_kv_scheme_name(wubu_kv_scheme_t kv) {
    switch (kv) {
        case WUBU_KV_F32:  return "F32";
        case WUBU_KV_F16:  return "F16";
        case WUBU_KV_Q4_0: return "Q4_0";
        case WUBU_KV_Q8:   return "OUR_Q8";
        case WUBU_KV_KIVI: return "KIVI";
        case WUBU_KV_VQ:   return "VQ";  /* KB2 doc 014 */
        default:           return "?";
    }
}

wubu_kv_choice_t wubu_kv_select(const wubu_roofline_cfg_t *cfg,
                                 double P_params, int B, int s) {
    wubu_kv_choice_t c;
    memset(&c, 0, sizeof(c));

    wubu_compress_target_t adv = wubu_roofline_advise(cfg, P_params, B, s);
    (void)wubu_roofline_bstar; /* bstar used by callers for logging only */

    if (adv == WUBU_COMPRESS_KV) {
        /* KV-dominated (B large vs B*). Pick KIVI (per-token V) when the
         * context is long -- that's where KIVI's asymmetric K!=V precision
         * earns its keep; otherwise our near-lossless Q8_0. */
        if (s >= 32768) {
            c.kv = WUBU_KV_KIVI;
            c.kv_bits = 8;
            c.why = "BW-bound, long ctx: KIVI per-token V";
        } else {
            c.kv = WUBU_KV_Q8;
            c.kv_bits = 8;
            c.why = "BW-bound: OUR_Q8 (near-lossless 2x vs fp16)";
        }
        c.weight_bits = cfg->bw_bits; /* keep weights as-is */
    } else if (adv == WUBU_COMPRESS_WEIGHTS) {
        /* W-dominated (B small vs B*): compress weights, keep KV at fp16. */
        c.kv = WUBU_KV_F16;
        c.kv_bits = 16;
        c.weight_bits = 4; /* recommend int4 weights */
        c.why = "weight-bound (B<B*): compress weights to int4, KV fp16";
    } else {
        c.kv = WUBU_KV_F16;
        c.kv_bits = 16;
        c.weight_bits = cfg->bw_bits;
        c.why = "degenerate: KV fp16, no compression";
    }
    return c;
}
