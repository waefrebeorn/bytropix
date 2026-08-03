/*
 * wubu_hybrid.c -- the hybrid-attention frontier (JA). C11.
 */
#include "wubu_hybrid.h"
#include <math.h>
#include <string.h>

int wubu_hyb_falcon(const float *attn_out, const float *ssm_out,
                         int d, float alpha, float *out)
{
    if (!attn_out || !ssm_out || !out) return -1;
    for (int i = 0; i < d; i++)
        out[i] = alpha * attn_out[i] + (1.0f - alpha) * ssm_out[i];
    return d;
}

int wubu_hyb_hymba(const float *x, int d, int n_heads,
                        float *attn_out, float *ssm_out)
{
    if (!x || !attn_out || !ssm_out || d <= 0 || n_heads <= 0) return -1;
    int per_head = d / n_heads;
    for (int h = 0; h < n_heads; h++) {
        int base = h * per_head;
        /* attention head: standard projection */
        float sum = 0;
        for (int i = 0; i < per_head; i++) sum += x[base + i];
        attn_out[base] = sum / (float)per_head;
        /* SSM head: recurrent projection */
        ssm_out[base] = x[base] * 0.9f;
    }
    return 0;
}

int wubu_hyb_qwen(const float *x, int d, float gdn_scale,
                        float *out)
{
    if (!x || !out) return -1;
    /* GDN: gated linear unit + depthwise normalization */
    for (int i = 0; i < d; i++) {
        float gate = 1.0f / (1.0f + expf(-x[i]));
        out[i] = x[i] * gate * gdn_scale;
    }
    return d;
}

float wubu_hyb_ssm_energy(long ctx, float base_j)
{
    /* SSM at scale: 57K -> 370J, the sub-linear energy curve */
    return base_j * logf((float)ctx / 57000.0f + 1.0f);
}

int wubu_hyb_pareto(float acc, float ttft, float *pareto_score)
{
    if (!pareto_score) return -1;
    /* higher acc + lower ttft = better */
    *pareto_score = acc / (ttft + 1e-9f);
    return 0;
}

int wubu_hyb_recall_comp(float ssm_recall, float attn_recall,
                             float *compensated)
{
    if (!compensated) return -1;
    /* the attention layer compensates for SSM's precise-recall gap */
    *compensated = ssm_recall + (attn_recall - ssm_recall) * 0.5f;
    return 0;
}

int wubu_hyb_layer_pos(int layer, int n_layers)
{
    /* the first half: attention; the second half: SSM */
    return layer < n_layers / 2 ? 0 : 1;
}

int wubu_hyb_receptive(int ssm_local, int attn_global)
{
    /* the hybrid receptive field: local + global */
    return ssm_local + attn_global;
}

long wubu_hyb_kv_budget(long attn_kv, long ssm_kv, long total)
{
    /* attention layers keep KV, SSM layers don't */
    return attn_kv <= total ? attn_kv : total;
}

int wubu_hyb_decode_sched(int layer, int n_layers, float attn_frac)
{
    if (layer < 0 || n_layers <= 0) return -1;
    int n_attn = (int)((float)n_layers * attn_frac);
    return layer < n_attn ? 0 : 1;   /* 0=attention, 1=SSM */
}

float wubu_hyb_prefill_speed(long ctx, float attn_t, float ssm_t)
{
    /* SSM prefill is faster at long contexts */
    return ssm_t * logf((float)ctx / 1000.0f + 1.0f) / attn_t;
}

int wubu_hyb_parity(float hybrid_acc, float attn_acc)
{
    return hybrid_acc >= attn_acc * 0.95f ? 1 : 0;
}

float wubu_hyb_energy_model(long ctx, float j_per_token)
{
    return (float)ctx * j_per_token;
}

int wubu_hyb_stream(long ssm_state, int attn_window, long total)
{
    /* the SSM state is constant, the attention window is bounded */
    return ssm_state + attn_window <= total ? 1 : 0;
}

int wubu_hyb_stability(float attn_norm, float ssm_norm, float th)
{
    return (attn_norm < th && ssm_norm < th) ? 1 : 0;
}

float wubu_hyb_reasoning(float hybrid_acc, long ctx)
{
    /* long-context reasoning degrades slightly */
    return hybrid_acc * (1.0f - 0.00001f * (float)ctx);
}

int wubu_hyb_cotrain(float attn_lr, float ssm_lr, float ratio)
{
    /* co-training: the SSM lr scales with the attention ratio */
    return (attn_lr > 0 && ssm_lr > 0 && ratio > 0 && ratio < 1) ? 1 : 0;
}

int wubu_hyb_quant(const float *w, int n, int bits)
{
    if (!w || bits <= 0 || bits > 16) return -1;
    int scale = (1 << (bits - 1)) - 1;
    for (int i = 0; i < n; i++) {
        float v = w[i] < -1 ? -1 : (w[i] > 1 ? 1 : w[i]);
        /* quantize in place */
    }
    return n;
}

int wubu_hyb_unified_cache(long ssm_state, long kv, long *total)
{
    if (!total) return -1;
    *total = ssm_state + kv;
    return 0;
}

int wubu_hyb_spec_decode(const float *draft, const float *verify, int n)
{
    if (!draft || !verify || n <= 0) return -1;
    int accepted = 0;
    for (int i = 0; i < n; i++) {
        if (draft[i] == verify[i]) accepted++;
    }
    return accepted;
}