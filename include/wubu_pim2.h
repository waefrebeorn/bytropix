/*
 * wubu_pim2.h -- the PIM/hardware frontier, complete (IS). C11.
 * Agnostic hardware-abstraction: a kernel-dispatch table + per-device
 * cost models, so the engine picks the mechanism by counters instead
 * of hardcoding. Covers CIM precision, endurance, near-memory decode,
 * heterogeneity, memory-wall budgets, near-storage, mapping, energy
 * ledgers, correctness audits, auto-tuning, rooflines, refresh, parity.
 */
#ifndef WUBU_PIM2_H
#define WUBU_PIM2_H

#include <stdint.h>

/* IS21: per-layer bit precision by sensitivity. */
int wubu_pim2_bits(float sensitivity, float th_lo, float th_hi);

/* IS22: endurance budget for KV writes. */
int wubu_pim2_endurance(long writes, long budget);

/* IS24: capacity-vs-latency frontier tradeoff. */
float wubu_pim2_frontier(long capacity, float latency);

/* IS25: heterogeneous scheduling (CPU/GPU/NPU-PIM). */
int wubu_pim2_hetero(const float *device_eff, int n, int *chosen);

/* IS26: memory-wall budget governor. */
int wubu_pim2_wall(long bytes, long cap_per_token);

/* IS27: integer KV for CIM. */
int wubu_pim2_int_kv(const float *kv, int n, int bits, int32_t *out);

/* IS28: near-storage RAG retrieval. */
int wubu_pim2_ns_rag(float retrieval_cost, float transfer_cost);

/* IS29: the kernel dispatch table. */
typedef struct { int op; int device; float cost; } wubu_pim2_dispatch_t;
int wubu_pim2_dispatch(const wubu_pim2_dispatch_t *table, int n, int op,
                       int *device);

/* IS30: crossbar placement optimizer. */
int wubu_pim2_place(int rows, int cols, int crossbar_h, int *n_arrays);

/* IS31: in-memory energy accounting. */
float wubu_pim2_energy(long ops, float j_per_op);

/* IS33: analog error bound audit. */
float wubu_pim2_audit(float adc_bits, float noise);

/* IS35: memory-centric decode organization. */
int wubu_pim2_mem_centric(float mem_locality, float th);

/* IS36: PIM-vs-CPU benefit predictor. */
int wubu_pim2_benefit(float pim_time, float cpu_time, float margin);

/* IS38: weight-stationary layout. */
int wubu_pim2_stationary(const float *w, int rows, int cols, float *out);

/* IS39: hardware counter model. */
long wubu_pim2_counters(long cycles, long bytes, long joules);

/* IS40: KV page movement between tiers. */
int wubu_pim2_page_move(float tier_from, float tier_to);

/* IS42: emerging-memory latency (PCM/FeFET). */
float wubu_pim2_latency(int tier, long bytes);

/* IS43: PIM-aware batching (shapes fitting the arrays). */
int wubu_pim2_batch_shape(long rows, long array_rows, long *batches);

/* IS45: low-precision accumulation guards. */
float wubu_pim2_acc_guard(float acc, float limit);

/* IS49: capacity planning (model + KV fit). */
int wubu_pim2_plan(long model, long kv, long device);

/* IS52: dataflow optimization (input vs output stationary). */
float wubu_pim2_dataflow(float reuse_in, float reuse_out);

/* IS53: energy roofline update. */
float wubu_pim2_roofline(long bytes, float j_per_byte);

/* IS55: CIM weight refresh (drift compensation). */
int wubu_pim2_refresh(float drift, float th);

/* IS56: per-device bit choice. */
int wubu_pim2_device_bits(float device_precision, float sens);

/* IS57: host-parity regression check. */
float wubu_pim2_parity(const float *host, const float *pim, int n);

/* IS58: near-memory top-k. */
int wubu_pim2_topk(const float *vals, int n, int k, int *idx);

/* IS60: the hardware diversity matrix. */
uint32_t wubu_pim2_matrix(int *devices, int n);

/* IS61: in-memory power cap. */
int wubu_pim2_powercap(float watts, float cap);

/* IS63: the hardware cost ledger. */
float wubu_pim2_ledger(long tokens, float j_per_token);

/* IS65: memory-centric planning. */
int wubu_pim2_plan_centric(float mem_fraction, float th);

#endif
