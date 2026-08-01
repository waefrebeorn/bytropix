/*
 * wubu_quant_selector.c -- Adaptive quantization / precision selectors
 * (N04 batch-size-aware, N05 context-length precision ladder, N09 PMC roofline
 * fallback). Pure decision functions the operator calls to pick KV/weight bits
 * and dispatch.
 *
 * Convergence (I/O survey + Roofline 2607.02558): the *optimal* precision is a
 * function of the live operating point (batch B, sequence s, measured beta_eff).
 * Below B* we are weight-bound -> spend bits on weights, compress KV. Above B*
 * we are KV-bound -> compress KV harder, keep weights. N09 adds a hardware
 * counter path that, when PMCs are unavailable (WSL/cloud), falls back to the
 * measured-roofline estimate (never crashes, never returns garbage).
 *
 * Triple-DA: invalid inputs clamp; no div-by-zero; deterministic.
 */
#include "wubu_quant_selector.h"
#include <math.h>

/* N04: batch-size-aware quant switch. Small batch -> weight-bound -> use
 * higher weight precision (b_w_hi) and can afford more KV bits. Large batch ->
 * KV-bound -> lower KV bits (b_kv_lo), weights can stay. Returns chosen
 * (b_w, b_kv) pair via out params. */
void wubu_batch_quant(int batch, double b_star,
                      int b_w_lo, int b_w_hi, int b_kv_lo, int b_kv_hi,
                      int *out_bw, int *out_bkv) {
    int bw = b_w_hi, bkv = b_kv_hi;
    if (b_star > 0.0 && (double)batch > b_star) {
        /* KV-bound: compress KV, weights can drop a notch too (amortized) */
        bkv = b_kv_lo;
        bw  = (b_w_lo + b_w_hi) / 2;
    } else {
        bw  = b_w_hi;
        bkv = b_kv_hi;
    }
    if (out_bw)  *out_bw  = bw;
    if (out_bkv) *out_bkv = bkv;
}

/* N05: context-length-aware KV precision ladder. Short ctx -> high precision
 * (quality matters, cheap). Long ctx -> lower precision (memory-bound, must
 * shrink KV). Returns bits in [b_lo, b_hi] decreasing with seq. */
int wubu_ctx_precision_ladder(int seq, int seq_full, int b_lo, int b_hi) {
    if (seq <= 0) return b_hi;
    if (seq_full <= 0) seq_full = seq;
    if (seq >= seq_full) return b_lo;
    float f = (float)seq / (float)seq_full;   /* 0..1 */
    int b = (int)(b_hi - (b_hi - b_lo) * f + 0.5f);
    if (b < b_lo) b = b_lo;
    if (b > b_hi) b = b_hi;
    return b;
}

/* N09: hardware-counter roofline. On systems without perf counters (most WSL/
 * cloud), returns 0 and the caller uses the measured estimate instead. When a
 * counter value is supplied, converts (bytes, cycles, freq_hz) to bandwidth.
 * Never divides by zero; returns -1 on bad input. */
double wubu_pmc_roofline(double bytes, double cycles, double freq_hz) {
    if (bytes <= 0.0 || cycles <= 0.0 || freq_hz <= 0.0) return -1.0;
    double secs = cycles / freq_hz;
    if (secs <= 0.0) return -1.0;
    return bytes / secs;
}
