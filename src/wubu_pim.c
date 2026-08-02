/*
 * wubu_pim.c -- processing-in-memory / near-memory frontier (Theme IS).
 */
#include "wubu_pim.h"
#include <math.h>
#include <string.h>

int wubu_pim_offload(int op_kind, long bytes, long compute_flops,
                     float mem_bw, float compute_bw)
{
    if (mem_bw <= 0 || compute_bw <= 0) return 0;
    /* memory-bound (bytes >> flops) + a GEMV-like op -> offload */
    float mem_time = (float)bytes / mem_bw;
    float cmp_time = (float)compute_flops / compute_bw;
    if (op_kind == 0 && bytes > 0 && mem_time > cmp_time) return 1;
    return 0;
}

int wubu_pim_gemv(const float *w, int rows, int cols, const float *v,
                  float *out, float adc_bits)
{
    if (!w || !v || !out || rows <= 0 || cols <= 0) return -1;
    float q = (adc_bits >= 8) ? 1.0f / 256.0f : 1.0f / (float)(1 << 6);
    for (int i = 0; i < rows; i++) {
        float acc = 0;
        for (int j = 0; j < cols; j++) acc += w[i * cols + j] * v[j];
        /* the ADC quantization + the noise floor */
        float err = acc * q * 0.5f;
        out[i] = acc + err;
    }
    return 0;
}

int wubu_pim_quant_gemv(const int8_t *w, int rows, int cols,
                        const int8_t *v, int32_t *out, int bits)
{
    if (!w || !v || !out || bits <= 0 || bits > 8) return -1;
    int32_t clip = (1 << (bits - 1)) - 1;
    for (int i = 0; i < rows; i++) {
        int64_t acc = 0;
        for (int j = 0; j < cols; j++) {
            int32_t wv = w[i * cols + j];
            if (wv > clip) wv = clip;
            if (wv < -clip - 1) wv = -clip - 1;
            acc += (int64_t)wv * v[j];
        }
        out[i] = (int32_t)acc;
    }
    return 0;
}

float wubu_pim_tier_cost(int tier, long bytes)
{
    /* tiers: 0 HBM, 1 3D-DRAM, 2 RRAM, 3 FeFET, 4 SOT-MRAM */
    static const float pJ_per_bit[5] = { 1.0f, 1.5f, 0.4f, 0.6f, 2.0f };
    if (tier < 0 || tier > 4) return -1;
    return pJ_per_bit[tier] * (float)bytes * 8.0f;
}

int wubu_pim_capacity(long pim_bytes, long model_bytes, float margin)
{
    if (margin <= 0) return 0;
    return pim_bytes >= (long)((float)model_bytes * margin) ? 1 : 0;
}

long wubu_pim_bytes_moved(int rows, int cols, int dtype_bytes)
{
    if (dtype_bytes <= 0) return -1;
    return (long)rows * cols * dtype_bytes;
}

int wubu_pim_dispatch(int is_gemv, long bytes, long flops,
                      float npu_eff, float pim_eff)
{
    if (is_gemv && bytes > flops && pim_eff > npu_eff) return 1; /* PIM */
    return 0;
}

int wubu_pim_layout(const float *w, int rows, int cols, float *out)
{
    if (!w || !out) return -1;
    /* channel-last: (c, r) so a row of the matrix is contiguous */
    for (int c = 0; c < cols; c++)
        for (int r = 0; r < rows; r++)
            out[c * rows + r] = w[r * cols + c];
    return rows * cols;
}

float wubu_pim_noise(float value, float adc_bits)
{
    float step = 1.0f / (1u << ((unsigned)adc_bits > 12 ? 12 : (unsigned)adc_bits));
    float err = step * 0.5f;
    return value + err;
}

float wubu_pim_op_cost(int op, long bytes, long flops,
                       float e_byte, float e_flop)
{
    (void)op;
    return (float)bytes * e_byte + (float)flops * e_flop;
}

int wubu_pim_batch(long *ops, int n, long threshold)
{
    if (!ops || n <= 0 || threshold <= 0) return -1;
    int batches = 0, i = 0;
    while (i < n) {
        long acc = 0;
        while (i < n && acc < threshold) acc += ops[i++];
        batches++;
    }
    return batches;
}

int wubu_pim_reduce(const float *partials, int n, float *sum)
{
    if (!partials || !sum) return -1;
    float s = 0;
    for (int i = 0; i < n; i++) s += partials[i];
    *sum = s;
    return 0;
}
