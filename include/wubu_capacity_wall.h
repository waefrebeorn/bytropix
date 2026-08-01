/*
 * wubu_capacity_wall.h -- Decode I/O capacity/roofline predictor (N12/N11).
 * Opaque-free: pure functions over decode-I/O math.
 */
#ifndef WUBU_CAPACITY_WALL_H
#define WUBU_CAPACITY_WALL_H

/* KV cache bytes for a GQA model at (L, n_kv, d_h, b_kv), batch B, seq s. */
double wubu_kv_cache_bytes(int L, int n_kv, int d_h, int b_kv,
                           int batch, int seq);

/* Weight bytes for `params` parameters at b_w bits. */
double wubu_weight_bytes(double params, int b_w);

/* 1 if KV cache for (batch, seq) fits in ram_bytes (the 512K-OOM gate). */
int wubu_kv_fits_ram(int L, int n_kv, int d_h, int b_kv,
                     int batch, int seq, double ram_bytes);

/* Crossover batch B* where KV-I/O overtakes weight-I/O. -1 if never KV-bound. */
double wubu_b_star(double weight_params, int b_w,
                   int L, int n_kv, int d_h, int b_kv, int seq);

/* Predicted time-per-output-token (seconds) for batch B at seq s. */
double wubu_tpot(double weight_params, int b_w,
                 int L, int n_kv, int d_h, int b_kv,
                 int batch, int seq, double beta_eff);

/* Tokens/second estimate (= 1 / TPOT). */
double wubu_tok_per_sec(double weight_params, int b_w,
                        int L, int n_kv, int d_h, int b_kv,
                        int batch, int seq, double beta_eff);

/* OOM-risk early-warning (N18): 1 if projected KV would exceed engage_frac of
 * ram (streaming/eviction should engage). */
int wubu_oom_risk(double weight_params, int b_w,
                  int L, int n_kv, int d_h, int b_kv,
                  int batch, int seq, double ram_bytes, double engage_frac);

/* N13 regime classifier: 0=WEIGHT_BOUND, 1=BALANCED, 2=KV_BOUND. */
int wubu_regime(double b_star, int batch, double tol);

#endif /* WUBU_CAPACITY_WALL_H */
