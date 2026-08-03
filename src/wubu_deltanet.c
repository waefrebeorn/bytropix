/*
 * wubu_deltanet.c -- the Gated-DeltaNet linear mixer (research 008).
 */
#include "wubu_deltanet.h"
#include <stdlib.h>
#include <string.h>

int wubu_deltanet_state_init(wubu_deltanet_state_t *st, int n_heads,
                             int head_dim)
{
    if (!st || n_heads <= 0 || head_dim <= 0) return -1;
    st->n_heads = n_heads;
    st->head_dim = head_dim;
    st->S = (float *)calloc((size_t)n_heads * head_dim * head_dim,
                            sizeof(float));
    if (!st->S) return -1;
    return 0;
}

void wubu_deltanet_state_reset(wubu_deltanet_state_t *st)
{
    if (!st || !st->S) return;
    memset(st->S, 0, (size_t)st->n_heads * st->head_dim * st->head_dim *
           sizeof(float));
}

void wubu_deltanet_read(wubu_deltanet_state_t *st, int head,
                        const float *k, float *out)
{
    if (!st || !st->S || !k || !out) return;
    int hd = st->head_dim;
    const float *S = st->S + (size_t)head * hd * hd;
    for (int i = 0; i < hd; i++) {
        float acc = 0;
        for (int j = 0; j < hd; j++) acc += S[i * hd + j] * k[j];
        out[i] = acc;
    }
}

void wubu_deltanet_step(wubu_deltanet_state_t *st, int head,
                        const float *k, const float *v,
                        float alpha, float beta, float *out)
{
    if (!st || !st->S || !k || !v || !out) return;
    int hd = st->head_dim;
    float *S = st->S + (size_t)head * hd * hd;
    /* Sk = S · k^T  (the current readout) */
    float Sk[256];
    for (int i = 0; i < hd; i++) {
        float acc = 0;
        for (int j = 0; j < hd; j++) acc += S[i * hd + j] * k[j];
        Sk[i] = acc;
    }
    /* the delta rule: S = α·S + β·k⊗(v − Sk) */
    for (int i = 0; i < hd; i++) {
        float err = v[i] - Sk[i];
        for (int j = 0; j < hd; j++)
            S[i * hd + j] = alpha * S[i * hd + j] + beta * k[j] * err;
    }
    /* the output = the NEW readout (o = S·k^T after the update) */
    if (out) {
        for (int i = 0; i < hd; i++) {
            float acc = 0;
            for (int j = 0; j < hd; j++) acc += S[i * hd + j] * k[j];
            out[i] = acc;
        }
    }
}

int wubu_deltanet_prefill(wubu_deltanet_state_t *st,
                          const float *K, const float *V,
                          int T, float alpha, float beta,
                          float *outs)
{
    if (!st || !K || !V || T <= 0 || !outs) return -1;
    int hd = st->head_dim, nh = st->n_heads;
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < nh; h++) {
            const float *k = K + ((size_t)t * nh + h) * hd;
            const float *v = V + ((size_t)t * nh + h) * hd;
            float *o = outs + ((size_t)t * nh + h) * hd;
            wubu_deltanet_step(st, h, k, v, alpha, beta, o);
        }
    }
    return 0;
}

void wubu_deltanet_state_free(wubu_deltanet_state_t *st)
{
    if (!st) return;
    free(st->S);
    st->S = NULL;
    st->n_heads = 0;
    st->head_dim = 0;
}
