/*
 * wubu_quant_selector.h -- Adaptive quant / precision selectors (N04/N05/N09).
 */
#ifndef WUBU_QUANT_SELECTOR_H
#define WUBU_QUANT_SELECTOR_H

/* N04 batch-size-aware quant switch. Writes chosen (b_w, b_kv) to out params. */
void wubu_batch_quant(int batch, double b_star,
                      int b_w_lo, int b_w_hi, int b_kv_lo, int b_kv_hi,
                      int *out_bw, int *out_bkv);

/* N05 context-length-aware KV precision ladder -> bits in [b_lo, b_hi]. */
int wubu_ctx_precision_ladder(int seq, int seq_full, int b_lo, int b_hi);

/* N09 hardware-counter roofline (bytes/cycles/freq). -1 on bad input. */
double wubu_pmc_roofline(double bytes, double cycles, double freq_hz);

#endif /* WUBU_QUANT_SELECTOR_H */
