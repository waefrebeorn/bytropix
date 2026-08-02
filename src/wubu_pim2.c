/*
 * wubu_pim2.c -- the PIM/hardware frontier, complete (IS). C11.
 */
#include "wubu_pim2.h"
#include <math.h>
#include <string.h>

int wubu_pim2_bits(float sensitivity, float th_lo, float th_hi)
{
    if (sensitivity > th_hi) return 8;   /* sensitive -> high precision */
    if (sensitivity > th_lo) return 4;
    return 2;
}

int wubu_pim2_endurance(long writes, long budget)
{
    if (budget <= 0) return 0;
    return writes <= budget ? 1 : 0;
}

float wubu_pim2_frontier(long capacity, float latency)
{
    /* the classic tradeoff: capacity grows, latency grows sub-linearly */
    return latency * (1.0f + 0.1f * logf((float)capacity + 1.0f));
}

int wubu_pim2_hetero(const float *device_eff, int n, int *chosen)
{
    if (!device_eff || !chosen || n <= 0) return -1;
    int best = 0;
    for (int i = 1; i < n; i++)
        if (device_eff[i] > device_eff[best]) best = i;
    *chosen = best;
    return 0;
}

int wubu_pim2_wall(long bytes, long cap_per_token)
{
    if (cap_per_token <= 0) return 0;
    return bytes <= cap_per_token ? 1 : 0;
}

int wubu_pim2_int_kv(const float *kv, int n, int bits, int32_t *out)
{
    if (!kv || !out || bits <= 0 || bits > 16) return -1;
    int32_t scale = (1 << (bits - 1)) - 1;
    for (int i = 0; i < n; i++) {
        float v = kv[i] < -1 ? -1 : (kv[i] > 1 ? 1 : kv[i]);
        out[i] = (int32_t)(v * scale);
    }
    return n;
}

int wubu_pim2_ns_rag(float retrieval_cost, float transfer_cost)
{
    return retrieval_cost < transfer_cost ? 1 : 0;
}

int wubu_pim2_dispatch(const wubu_pim2_dispatch_t *table, int n, int op,
                       int *device)
{
    if (!table || !device || n <= 0) return -1;
    int found = -1;
    float best = 1e30f;
    for (int i = 0; i < n; i++) {
        if (table[i].op == op && table[i].cost < best) {
            best = table[i].cost; found = i;
        }
    }
    if (found < 0) return -1;
    *device = table[found].device;
    return 0;
}

int wubu_pim2_place(int rows, int cols, int crossbar_h, int *n_arrays)
{
    if (!n_arrays || crossbar_h <= 0 || rows <= 0 || cols <= 0) return -1;
    int arrays_r = (rows + crossbar_h - 1) / crossbar_h;
    *n_arrays = arrays_r * cols;
    return 0;
}

float wubu_pim2_energy(long ops, float j_per_op)
{
    return (float)ops * j_per_op;
}

float wubu_pim2_audit(float adc_bits, float noise)
{
    /* the error bound: the quantization step + the noise floor */
    float step = 1.0f / (float)(1 << (adc_bits > 12 ? 12 : (int)adc_bits));
    return step * 0.5f + noise;
}

int wubu_pim2_mem_centric(float mem_locality, float th)
{
    return mem_locality >= th ? 1 : 0;
}

int wubu_pim2_benefit(float pim_time, float cpu_time, float margin)
{
    return pim_time < cpu_time * (1.0f - margin) ? 1 : 0;
}

int wubu_pim2_stationary(const float *w, int rows, int cols, float *out)
{
    if (!w || !out) return -1;
    /* weight-stationary: the weights stay fixed, the activations stream.
     * The layout is the natural row-major (no transpose needed). */
    memcpy(out, w, sizeof(float) * rows * cols);
    return rows * cols;
}

long wubu_pim2_counters(long cycles, long bytes, long joules)
{
    return cycles + bytes + joules * 1000;
}

int wubu_pim2_page_move(float tier_from, float tier_to)
{
    return tier_to < tier_from ? 1 : 0;
}

float wubu_pim2_latency(int tier, long bytes)
{
    /* tiers: 0 HBM, 1 3D-DRAM, 2 RRAM, 3 FeFET, 4 PCM */
    static const float ns_per_byte[5] = { 0.1f, 0.3f, 1.0f, 1.2f, 2.0f };
    if (tier < 0 || tier > 4) return -1;
    return ns_per_byte[tier] * (float)bytes;
}

int wubu_pim2_batch_shape(long rows, long array_rows, long *batches)
{
    if (!batches || array_rows <= 0) return -1;
    *batches = (rows + array_rows - 1) / array_rows;
    return 0;
}

float wubu_pim2_acc_guard(float acc, float limit)
{
    if (acc > limit) return limit;
    if (acc < -limit) return -limit;
    return acc;
}

int wubu_pim2_plan(long model, long kv, long device)
{
    return (model + kv) <= device ? 1 : 0;
}

float wubu_pim2_dataflow(float reuse_in, float reuse_out)
{
    /* input-stationary wins when the input reuses more */
    return reuse_in - reuse_out;
}

float wubu_pim2_roofline(long bytes, float j_per_byte)
{
    return (float)bytes * j_per_byte;
}

int wubu_pim2_refresh(float drift, float th)
{
    return drift >= th ? 1 : 0;
}

int wubu_pim2_device_bits(float device_precision, float sens)
{
    if (sens > 0.7f) return 8;
    return device_precision >= 0.8f ? 8 : 4;
}

float wubu_pim2_parity(const float *host, const float *pim, int n)
{
    if (!host || !pim || n <= 0) return 1.0f;
    float max_err = 0;
    for (int i = 0; i < n; i++) {
        float e = fabsf(host[i] - pim[i]);
        if (e > max_err) max_err = e;
    }
    return max_err;
}

int wubu_pim2_topk(const float *vals, int n, int k, int *idx)
{
    if (!vals || !idx || k <= 0 || n <= 0) return -1;
    if (k > n) k = n;
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (vals[idx[j]] > vals[idx[i]]) {
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }
    return k;
}

uint32_t wubu_pim2_matrix(int *devices, int n)
{
    uint32_t m = 0;
    for (int i = 0; i < n; i++)
        if (devices[i] >= 0 && devices[i] < 32) m |= (1u << devices[i]);
    return m;
}

int wubu_pim2_powercap(float watts, float cap)
{
    return watts <= cap ? 1 : 0;
}

float wubu_pim2_ledger(long tokens, float j_per_token)
{
    return (float)tokens * j_per_token;
}

int wubu_pim2_plan_centric(float mem_fraction, float th)
{
    return mem_fraction >= th ? 1 : 0;
}
