/*
 * wubu_roofline.c — Roofline / B*-crossover auto-tuner (Round-2 item #101).
 * C11, self-contained. Implements the data-movement framework from the I/O
 * survey: given model params + weight/KV precision + batch + context, decide
 * whether the system is weight- (W-) or KV-cache- (K-) dominated, and auto
 * pick the compression target. This is the meta-analysis's #1 high-leverage win.
 *
 * Key formula (survey eq. B2):
 *   B* = P * bw / (2 * L * s * n_kv * d_h * bkv/8)
 *   B < B*  -> W-dominated -> compress WEIGHTS
 *   B > B*  -> K-dominated  -> compress KV
 */
#include "wubu_roofline.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

wubu_roofline_cfg_t wubu_roofline_default(void) {
    wubu_roofline_cfg_t c;
    memset(&c, 0, sizeof(c));
    c.n_layers = 80;
    c.n_kv_heads = 8;        /* GQA */
    c.head_dim = 128;
    c.bw_bits = 16;          /* FP16 weights */
    c.bkv_bits = 16;         /* FP16 KV */
    c.beta_eff_tb_s = 2.68;  /* H100 effective HBM BW (TB/s) */
    return c;
}

/* Compute per-step weight I/O (GB) and KV I/O (GB) at batch B, context s. */
void wubu_roofline_io(const wubu_roofline_cfg_t *c, double P_params,
                      int B, int s, double *W_gb, double *K_gb) {
    *W_gb = P_params * (c->bw_bits / 8.0) / 1e9;            /* static per step */
    double Kseq = (double)c->n_layers * 2.0 * c->n_kv_heads * c->head_dim
                  * (c->bkv_bits / 8.0) * s / 1e9;          /* GB per sequence */
    *K_gb = (double)B * Kseq;
}

/* Crossover batch B* where K overtakes W. Returns -1 if degenerate. */
double wubu_roofline_bstar(const wubu_roofline_cfg_t *c, double P_params, int s) {
    double Kseq = (double)c->n_layers * 2.0 * c->n_kv_heads * c->head_dim
                  * (c->bkv_bits / 8.0) * s / 1e9;
    if (Kseq <= 0) return -1;
    double W = P_params * (c->bw_bits / 8.0) / 1e9;
    return W / Kseq;
}

/* Decision: which flow to compress next. */
wubu_compress_target_t wubu_roofline_advise(const wubu_roofline_cfg_t *c,
                                            double P_params, int B, int s) {
    double bstar = wubu_roofline_bstar(c, P_params, s);
    if (bstar < 0) return WUBU_COMPRESS_NONE;
    if (B < bstar) return WUBU_COMPRESS_WEIGHTS;
    return WUBU_COMPRESS_KV;
}

/* Estimated TPOT (ms) = (W + K) / beta_eff. */
double wubu_roofline_tpot_ms(const wubu_roofline_cfg_t *c, double P_params,
                             int B, int s) {
    double W, K;
    wubu_roofline_io(c, P_params, B, s, &W, &K);
    double bytes = (W + K) * 1e9;                 /* back to bytes */
    double sec = bytes / (c->beta_eff_tb_s * 1e12);
    return sec * 1000.0;
}
