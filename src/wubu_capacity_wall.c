/*
 * wubu_capacity_wall.c -- Decode I/O capacity/roofline predictor (N12/N11).
 *
 * Convergence (I/O survey 2026 + Roofline 2607.02558): at decode, the
 * binding constraint oscillates between weight-I/O (W) and KV-I/O (K) and
 * ultimately compute, as context/batch grow. These two predictors turn that
 * into numbers the operator can act on:
 *   - capacity_wall: does the KV cache for (B, s) fit in `ram_bytes`?
 *   - tpot:         predicted time-per-output-token = (W + K) / beta_eff.
 *   - b_star:       the crossover batch where K overtakes W (KV-bound past it).
 * Pure C11, no third-party dep, triple-DA edge cases (cap==0, s==0, bits).
 */
#include "wubu_capacity_wall.h"
#include <math.h>

/* KV bytes per token per layer for a GQA model with n_kv heads of dim d_h,
 * stored at b_kv bits. Returns bytes (not bits). */
static double kv_bytes_per_tok_per_layer(int n_kv, int d_h, int b_kv) {
    if (n_kv <= 0 || d_h <= 0 || b_kv <= 0) return 0.0;
    return (double)(2 * n_kv * d_h) * (b_kv / 8.0);
}

double wubu_kv_cache_bytes(int L, int n_kv, int d_h, int b_kv,
                           int batch, int seq) {
    if (seq <= 0 || batch <= 0) return 0.0;
    double per = kv_bytes_per_tok_per_layer(n_kv, d_h, b_kv);
    return per * L * batch * seq;
}

double wubu_weight_bytes(double params, int b_w) {
    if (params <= 0 || b_w <= 0) return 0.0;
    return params * (b_w / 8.0);
}

/* True if the KV cache for (batch, seq) fits in ram_bytes. This is the
 * 512K-OOM gate: if false, streaming/eviction must engage. */
int wubu_kv_fits_ram(int L, int n_kv, int d_h, int b_kv,
                     int batch, int seq, double ram_bytes) {
    if (ram_bytes <= 0) return 0;
    return wubu_kv_cache_bytes(L, n_kv, d_h, b_kv, batch, seq) <= ram_bytes;
}

/* Crossover batch B* where KV-I/O overtakes weight-I/O during decode.
 * B* = W_bytes / KV_bytes_per_seq. Below it -> weight-bound; above -> KV-bound.
 * Returns -1 if the model is never KV-bound (KV bytes/seq == 0). */
double wubu_b_star(double weight_params, int b_w,
                   int L, int n_kv, int d_h, int b_kv, int seq) {
    double W = wubu_weight_bytes(weight_params, b_w);
    double Kseq = kv_bytes_per_tok_per_layer(n_kv, d_h, b_kv) * L * seq;
    if (Kseq <= 0.0) return -1.0;
    return W / Kseq;
}

/* Predicted TPOT (seconds/token) for batch B at sequence s.
 * TPOT = (W_bytes + B*K_seq_bytes) / beta_eff. Uses W once + K per token. */
double wubu_tpot(double weight_params, int b_w,
                 int L, int n_kv, int d_h, int b_kv,
                 int batch, int seq, double beta_eff) {
    if (beta_eff <= 0) return 0.0;
    double W = wubu_weight_bytes(weight_params, b_w);
    double Kseq = kv_bytes_per_tok_per_layer(n_kv, d_h, b_kv) * L * seq;
    double total = W + (double)batch * Kseq;   /* bytes moved this decode step */
    return total / beta_eff;
}

/* Tokens/second estimate (= 1 / TPOT). */
double wubu_tok_per_sec(double weight_params, int b_w,
                        int L, int n_kv, int d_h, int b_kv,
                        int batch, int seq, double beta_eff) {
    double t = wubu_tpot(weight_params, b_w, L, n_kv, d_h, b_kv,
                         batch, seq, beta_eff);
    if (t <= 0.0) return 0.0;
    return 1.0 / t;
}

/* OOM-risk early-warning (N18): should streaming/eviction engage now?
 * Returns 1 if the projected KV cache for (batch, seq) would exceed
 * engage_frac of ram_bytes (default 0.9). The operator calls this each
 * step; when it trips, it enables StreamingKV/H2O so the cache never OOMs. */
int wubu_oom_risk(double weight_params, int b_w,
                  int L, int n_kv, int d_h, int b_kv,
                  int batch, int seq, double ram_bytes, double engage_frac) {
     if (ram_bytes <= 0) return 1;
     if (engage_frac <= 0.0) engage_frac = 0.9;
     double kv = wubu_kv_cache_bytes(L, n_kv, d_h, b_kv, batch, seq);
     /* include weights in the footprint only if not already resident-pinned */
     double footprint = kv; /* weights are amortized/streamed; KV is the live risk */
     return footprint > engage_frac * ram_bytes;
 }

 /* N13 compute-vs-bandwidth regime classifier. Given B* crossover, the
  * operating regime is:
  *   b_star >> batch  -> WEIGHT_BOUND (decode gated by weight I/O)
  *   b_star ~  batch  -> BALANCED
  *   b_star << batch  -> KV_BOUND    (decode gated by KV I/O)
  * Returns 0=WEIGHT, 1=BALANCED, 2=KV. Used by the operator to pick the lever
  * (compress KV vs. fuse weights vs. raise batch). */
 int wubu_regime(double b_star, int batch, double tol) {
     if (b_star < 0.0) return 2;          /* never weight-bound (no KV bytes) */
     if (tol <= 0.0) tol = 1.5;
     double r = b_star / (double)(batch > 0 ? batch : 1);
     if (r > tol) return 0;               /* weight-bound */
     if (r < 1.0 / tol) return 2;         /* kv-bound */
     return 1;                            /* balanced */
 }
