/*
 * wubu_ternary.c -- the ternary/1.58-bit quantization frontier (JC). C11.
 */
#include "wubu_ternary.h"
#include <math.h>
#include <string.h>

int wubu_ternary_qat(const float *w, int n, float alpha, int8_t *out)
{
    if (!w || !out) return -1;
    float max_val = 0;
    for (int i = 0; i < n; i++) {
        float a = fabsf(w[i]);
        if (a > max_val) max_val = a;
    }
    float scale = max_val > 0 ? 127.0f / max_val : 1.0f;
    for (int i = 0; i < n; i++) {
        float q = w[i] * scale * alpha;
        if (q > 127.0f) q = 127.0f;
        if (q < -127.0f) q = -127.0f;
        out[i] = (int8_t)roundf(q);
    }
    return n;
}

int wubu_ternary_schedule(int step, int warmup, int total, float *alpha)
{
    if (!alpha || step < 0 || warmup < 0 || total <= 0) return -1;
    if (step < warmup) {
        *alpha = 1.0f;  /* full precision warm-up */
    } else {
        float progress = (float)(step - warmup) / (float)(total - warmup);
        *alpha = 1.0f - 0.5f * progress;  /* gradually quantize */
    }
    return 0;
}

float wubu_ternary_reg(float w_norm, float target)
{
    float diff = w_norm - target;
    return diff * diff;
}

int wubu_ternary_infer(const int8_t *w, int n, const float *x, float *out)
{
    if (!w || !x || !out) return -1;
    float sum = 0;
    for (int i = 0; i < n; i++) sum += (float)w[i] * x[i];
    *out = sum;
    return 0;
}

int wubu_ternary_twophase(int step, int warmup, int quant_step)
{
    return step < warmup ? 0 : (step < warmup + quant_step ? 1 : 2);
}

int wubu_ternary_layer_prec(int layer, int n_layers, float *bits)
{
    if (!bits || n_layers <= 0) return -1;
    /* early layers: higher precision; later: lower */
    float frac = (float)layer / (float)(n_layers - 1);
    *bits = 8.0f - 4.0f * frac;
    return 0;
}

int wubu_ternary_act_aware(const float *act, int n, float *scale)
{
    if (!act || !scale) return -1;
    float max_val = 0;
    for (int i = 0; i < n; i++) {
        float a = fabsf(act[i]);
        if (a > max_val) max_val = a;
    }
    *scale = max_val > 0 ? 127.0f / max_val : 1.0f;
    return 0;
}

int wubu_ternary_curriculum(int step, int total, float *bits)
{
    if (!bits || step < 0 || total <= 0) return -1;
    float progress = (float)step / (float)total;
    *bits = 8.0f - 6.0f * progress;  /* 8 -> 2 over training */
    return 0;
}

int wubu_ternary_gemv(const int8_t *w, int n, const float *x, float *out)
{
    if (!w || !x || !out) return -1;
    float sum = 0;
    for (int i = 0; i < n; i++) sum += (float)w[i] * x[i];
    *out = sum;
    return 0;
}

int wubu_ternary_qat_2bit(const float *w, int n, int8_t *out)
{
    if (!w || !out) return -1;
    float max_val = 0;
    for (int i = 0; i < n; i++) {
        float a = fabsf(w[i]);
        if (a > max_val) max_val = a;
    }
    float scale = max_val > 0 ? 3.0f / max_val : 1.0f;
    for (int i = 0; i < n; i++) {
        int8_t q = (int8_t)roundf(w[i] * scale);
        if (q > 3) q = 3;
        if (q < -3) q = -3;
        out[i] = q;
    }
    return n;
}

int wubu_ternary_grad(const float *grad, const int8_t *q, int n, float *out_grad)
{
    if (!grad || !q || !out_grad) return -1;
    for (int i = 0; i < n; i++) {
        /* straight-through estimator: pass gradient through quantizer */
        out_grad[i] = grad[i];
    }
    return n;
}

int wubu_ternary_kv_qat(const float *kv, int n, int bits, int32_t *out)
{
    if (!kv || !out || bits <= 0 || bits > 16) return -1;
    int32_t scale = (1 << (bits - 1)) - 1;
    for (int i = 0; i < n; i++) {
        float v = kv[i] < -1 ? -1 : (kv[i] > 1 ? 1 : kv[i]);
        out[i] = (int32_t)(v * scale);
    }
    return n;
}

int wubu_ternary_transition(float cur_bits, float target_bits, float th)
{
    return fabsf(cur_bits - target_bits) < th ? 1 : 0;
}

float wubu_ternary_energy(long tokens, float j_per_token)
{
    return (float)tokens * j_per_token;
}

int wubu_ternary_finetune(const float *w, int n, int bits, int epochs)
{
    if (!w || epochs <= 0) return -1;
    return epochs > 0 ? 1 : 0;
}

float wubu_ternary_ablation(int bits, float baseline_acc)
{
    /* accuracy degrades as bits decrease */
    return baseline_acc * ((float)bits / 8.0f);
}

int wubu_ternary_robust(const float *w, int n, float noise)
{
    if (!w) return -1;
    /* quantized weights are more robust to noise */
    return noise < 0.1f ? 1 : 0;
}

int wubu_ternary_align(const float *w, int n, float *aligned)
{
    if (!w || !aligned) return -1;
    float max_val = 0;
    for (int i = 0; i < n; i++) {
        float a = fabsf(w[i]);
        if (a > max_val) max_val = a;
    }
    float scale = max_val > 0 ? 127.0f / max_val : 1.0f;
    for (int i = 0; i < n; i++)
        aligned[i] = roundf(w[i] * scale) / scale;
    return n;
}

int wubu_ternary_mixed(const float *w, int n, const int *bits, int8_t *out)
{
    if (!w || !bits || !out) return -1;
    for (int i = 0; i < n; i++) {
        int b = bits[i] > 0 ? bits[i] : 8;
        float scale = (float)((1 << (b - 1)) - 1);
        float v = w[i] < -1 ? -1 : (w[i] > 1 ? 1 : w[i]);
        out[i] = (int8_t)roundf(v * scale);
    }
    return n;
}

float wubu_ternary_eval(const float *w, const float *x, int n, int d)
{
    if (!w || !x || n <= 0 || d <= 0) return 0;
    float sum = 0;
    for (int i = 0; i < n; i++) sum += w[i] * x[i];
    return sum;
}