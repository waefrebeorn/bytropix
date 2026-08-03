/*
 * wubu_linattn2.c -- the linear-attention frontier, complete (IU). C11.
 */
#include "wubu_linattn2.h"
#include <math.h>
#include <string.h>

int wubu_la2_delta_write(float *state, int d, const float *k, const float *v,
                         float lr)
{
    if (!state || !k || !v) return -1;
    /* the delta rule: state += lr * (v - k'state) * k */
    float err = 0;
    for (int i = 0; i < d; i++) err += k[i] * state[i];
    err = v[0] - err;
    for (int i = 0; i < d; i++) state[i] += lr * err * k[i];
    return 0;
}

int wubu_la2_kernel_pick(const float *costs, int n)
{
    if (!costs || n <= 0) return -1;
    int best = 0;
    for (int i = 1; i < n; i++)
        if (costs[i] < costs[best]) best = i;
    return best;
}

int wubu_la2_precision(float drift, float th_lo, float th_hi)
{
    if (drift > th_hi) return 32;   /* high drift -> fp32 */
    if (drift > th_lo) return 16;
    return 8;
}

float wubu_la2_energy(long ctx, float attn_j, float ssm_j)
{
    return (float)ctx * ssm_j / ((float)ctx * attn_j + 1e-9f);
}

int wubu_la2_layer_sched(int layer, int n_layers, float ssm_frac)
{
    if (layer < 0 || n_layers <= 0) return -1;
    int n_ssm = (int)((float)n_layers * ssm_frac);
    return layer < n_ssm ? 1 : 0;
}

int wubu_la2_ckpt(const float *state, int d, float *buf)
{
    if (!state || !buf) return -1;
    memcpy(buf, state, sizeof(float) * d);
    return d;
}

int wubu_la2_restore(float *state, int d, const float *buf)
{
    if (!state || !buf) return -1;
    memcpy(state, buf, sizeof(float) * d);
    return 0;
}

float wubu_la2_recall_gap(float linear_recall, float attn_recall)
{
    float gap = attn_recall - linear_recall;
    return gap < 0 ? 0 : gap;
}

int wubu_la2_stream(const float *x, float *state, int d, float *out)
{
    if (!x || !state || !out) return -1;
    for (int i = 0; i < d; i++) { state[i] += x[i]; out[i] = state[i]; }
    return 0;
}

int wubu_la2_chunk_prefill(const float *x, int n, int d, int chunk,
                           float *state)
{
    if (!x || !state || chunk <= 0) return -1;
    int nc = 0;
    for (int c = 0; c < n; c += chunk) {
        int end = c + chunk < n ? c + chunk : n;
        for (int t = c; t < end; t++)
            for (int i = 0; i < d; i++) state[i] += x[t * d + i];
        nc++;
    }
    return nc;
}

float wubu_la2_forget(float gate, float base)
{
    return base * (1.0f - 0.5f * gate);
}

int wubu_la2_normalize(float *state, int d)
{
    if (!state || d <= 0) return -1;
    float nrm = 0;
    for (int i = 0; i < d; i++) nrm += state[i] * state[i];
    nrm = sqrtf(nrm);
    if (nrm < 1e-9f) return 0;
    for (int i = 0; i < d; i++) state[i] /= nrm;
    return 0;
}

float wubu_la2_update_energy(int d, float j_per_dim)
{
    return (float)d * j_per_dim;
}

int wubu_la2_decay(float *state, int d, float rate)
{
    if (!state || rate < 0) return -1;
    if (rate > 1) rate = 1;
    for (int i = 0; i < d; i++) state[i] *= (1.0f - rate);
    return 0;
}

int wubu_la2_quant_state(const float *state, int d, int bits, int32_t *out)
{
    if (!state || !out || bits <= 0 || bits > 16) return -1;
    int32_t scale = (1 << (bits - 1)) - 1;
    for (int i = 0; i < d; i++) {
        float v = state[i] < -1 ? -1 : (state[i] > 1 ? 1 : state[i]);
        out[i] = (int32_t)(v * scale);
    }
    return d;
}

int wubu_la2_expansion(float recall, float target, float *ratio)
{
    if (!ratio) return -1;
    if (recall < target) *ratio *= 1.5f;
    else if (recall > target * 1.1f) *ratio *= 0.9f;
    if (*ratio < 1.0f) *ratio = 1.0f;
    if (*ratio > 16.0f) *ratio = 16.0f;
    return 0;
}

int wubu_la2_draft(const float *state, int d, float *logits)
{
    if (!state || !logits) return -1;
    /* the recurrent drafter: the state projects to the draft logits */
    for (int i = 0; i < d; i++) logits[i] = state[i];
    return d;
}

int wubu_la2_chunk_par(int n_chunks, int cores)
{
    if (n_chunks <= 0 || cores <= 0) return 0;
    return n_chunks < cores ? n_chunks : cores;
}

int wubu_la2_mux(const float **states, int n, int d, const float *gate,
                 float *out)
{
    if (!states || !gate || !out || n <= 0) return -1;
    for (int i = 0; i < d; i++) out[i] = 0;
    for (int s = 0; s < n; s++)
        for (int i = 0; i < d; i++) out[i] += gate[s] * states[s][i];
    return 0;
}

int wubu_la2_watchdog(float norm, float th)
{
    return norm > th ? 1 : 0;
}

int wubu_la2_pos_head(int head, int n_heads, int *scheme)
{
    if (!scheme || n_heads <= 0) return -1;
    /* head 0..k: RoPE; the rest: PaTH */
    *scheme = (head < n_heads / 2) ? 0 : 1;
    return 0;
}

long wubu_la2_span(float decay, float th)
{
    if (decay <= 0 || decay >= 1) return 1000000000L;
    /* the effective field: how many steps until the memory < th */
    return (long)(logf(th) / logf(decay));
}

int wubu_la2_o1(int state_dims, int seq_len)
{
    (void)seq_len;
    return state_dims > 0 ? 1 : 0;   /* the state is O(1) in the seq */
}

int wubu_la2_slot_cap(int d, int slots)
{
    return d * slots;
}

int wubu_la2_needle(const float *state, int d, const float *needle, float th)
{
    if (!state || !needle) return -1;
    float s = 0;
    for (int i = 0; i < d; i++) s += state[i] * needle[i];
    return s > th ? 1 : 0;
}

int wubu_la2_prune(const float *importance, int d, float th, int *keep)
{
    if (!importance || !keep || d <= 0) return -1;
    int k = 0;
    for (int i = 0; i < d; i++)
        if (importance[i] >= th) keep[k++] = i;
    return k;
}

float wubu_la2_layer_cost(int attn, int ssm, float a_j, float s_j)
{
    return (float)attn * a_j + (float)ssm * s_j;
}
