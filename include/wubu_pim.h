/*
 * wubu_pim.h -- processing-in-memory / near-memory frontier (Theme IS).
 * C11. The hardware-abstracted PIM model: offload decisions, near-memory
 * tiers, crossbar GEMV emulation, quantization constraints, emerging
 * memory tiers, near-storage compute, data-movement accounting, hybrid
 * NPU-PIM dispatch, analog noise, cost-model integration, tiling.
 */
#ifndef WUBU_PIM_H
#define WUBU_PIM_H

#include <stdint.h>

/* IS01: PIM offload decision -- GEMV-over-KV goes near memory. */
int wubu_pim_offload(int op_kind, long bytes, long compute_flops,
                     float mem_bw, float compute_bw);

/* IS03: crossbar GEMV emulation (the analog MAC model). */
int wubu_pim_gemv(const float *w, int rows, int cols, const float *v,
                  float *out, float adc_bits);

/* IS04: SRAM-CIM bit-cell precision limits (quantized MAC). */
int wubu_pim_quant_gemv(const int8_t *w, int rows, int cols,
                        const int8_t *v, int32_t *out, int bits);

/* IS05: emerging-memory tier energy/latency model. */
float wubu_pim_tier_cost(int tier, long bytes);

/* IS07: PIM capacity wall guard. */
int wubu_pim_capacity(long pim_bytes, long model_bytes, float margin);

/* IS10: data-movement accounting (bytes moved). */
long wubu_pim_bytes_moved(int rows, int cols, int dtype_bytes);

/* IS09: hybrid NPU-PIM dispatch. */
int wubu_pim_dispatch(int is_gemv, long bytes, long flops,
                      float npu_eff, float pim_eff);

/* IS12: channel-last weight layout for in-memory MAC. */
int wubu_pim_layout(const float *w, int rows, int cols, float *out);

/* IS13: analog-compute noise (ADC/DAC quantization error). */
float wubu_pim_noise(float value, float adc_bits);

/* IS14: cost-model integration (energy + latency per op). */
float wubu_pim_op_cost(int op, long bytes, long flops,
                       float e_byte, float e_flop);

/* IS15: PIM offload batching (coalesce ops). */
int wubu_pim_batch(long *ops, int n, long threshold);

/* IS18: near-memory reduce (partial sums at the memory). */
int wubu_pim_reduce(const float *partials, int n, float *sum);

#endif
