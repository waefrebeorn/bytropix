/*
 * wubu_moe2.c -- the mixed-agents router (phase 2 of the WuBu model).
 */
#include "wubu_moe2.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int wubu_moe2_route(const wubu_moe2_t *moe, const float *x,
                    int active_idx[MOE2_N_ACTIVE],
                    float active_w[MOE2_N_ACTIVE])
{
    if (!moe || !moe->router_w || !x || !active_idx || !active_w) return -1;
    float scores[MOE2_N_EXPERTS];
    /* scores = x @ router_w, then softmax */
    float maxv = -1e30f;
    for (int e = 0; e < MOE2_N_EXPERTS; e++) {
        float acc = 0;
        const float *rw = moe->router_w + (size_t)e * MOE2_D_MODEL;
        for (int d = 0; d < MOE2_D_MODEL; d++) acc += rw[d] * x[d];
        scores[e] = acc;
        if (acc > maxv) maxv = acc;
    }
    double sum = 0;
    for (int e = 0; e < MOE2_N_EXPERTS; e++) {
        scores[e] = expf(scores[e] - maxv);
        sum += scores[e];
    }
    for (int e = 0; e < MOE2_N_EXPERTS; e++) scores[e] /= (float)sum;
    /* pick the top-k (insertion sort on indices) */
    int order[MOE2_N_EXPERTS];
    for (int e = 0; e < MOE2_N_EXPERTS; e++) order[e] = e;
    for (int i = 1; i < MOE2_N_EXPERTS; i++) {
        int j = i;
        while (j > 0 && scores[order[j]] > scores[order[j - 1]]) {
            int t = order[j]; order[j] = order[j - 1]; order[j - 1] = t;
            j--;
        }
    }
    for (int k = 0; k < MOE2_N_ACTIVE; k++) {
        active_idx[k] = order[k];
        active_w[k] = scores[order[k]];
    }
    return MOE2_N_ACTIVE;
}

int wubu_moe2_forward(const wubu_moe2_t *moe, const float *x, float *out)
{
    if (!moe || !x || !out) return -1;
    /* the shared expert: always on */
    if (moe->shared_gate && moe->shared_up && moe->shared_down) {
        float g[MOE2_D_FF], up[MOE2_D_FF];
        for (int d = 0; d < MOE2_D_FF; d++) {
            float acc = 0;
            const float *gw = moe->shared_gate + (size_t)d * MOE2_D_MODEL;
            for (int i = 0; i < MOE2_D_MODEL; i++) acc += gw[i] * x[i];
            g[d] = acc;
        }
        for (int d = 0; d < MOE2_D_FF; d++) {
            float acc = 0;
            const float *uw = moe->shared_up + (size_t)d * MOE2_D_MODEL;
            for (int i = 0; i < MOE2_D_MODEL; i++) acc += uw[i] * x[i];
            up[d] = acc;
        }
        float gate_scale = 0;
        if (moe->shared_gate_w) {
            for (int i = 0; i < MOE2_D_MODEL; i++) gate_scale += moe->shared_gate_w[i] * x[i];
            gate_scale = 1.0f / (1.0f + expf(-gate_scale));   /* sigmoid */
        } else gate_scale = 1.0f;
        for (int o = 0; o < MOE2_D_MODEL; o++) {
            float acc = 0;
            const float *dw = moe->shared_down + (size_t)o * MOE2_D_FF;
            for (int d = 0; d < MOE2_D_FF; d++)
                acc += dw[d] * (up[d] * (g[d] > 0 ? 1.0f : 0.0f));  /* swish-ish gate */
            out[o] = gate_scale * acc;
        }
    } else {
        memset(out, 0, MOE2_D_MODEL * sizeof(float));
    }

    /* the routed experts */
    int idx[MOE2_N_ACTIVE];
    float w[MOE2_N_ACTIVE];
    if (wubu_moe2_route(moe, x, idx, w) == MOE2_N_ACTIVE) {
        float contrib[MOE2_D_MODEL];
        for (int k = 0; k < MOE2_N_ACTIVE; k++) {
            int e = idx[k];
            float *gate = moe->exp_gate[e];
            float *up = moe->exp_up[e];
            float *down = moe->exp_down[e];
            if (!gate || !up || !down) continue;
            float g[MOE2_D_FF], u[MOE2_D_FF];
            for (int d = 0; d < MOE2_D_FF; d++) {
                float acc_g = 0, acc_u = 0;
                const float *gw = gate + (size_t)d * MOE2_D_MODEL;
                const float *uw = up + (size_t)d * MOE2_D_MODEL;
                for (int i = 0; i < MOE2_D_MODEL; i++) {
                    acc_g += gw[i] * x[i];
                    acc_u += uw[i] * x[i];
                }
                g[d] = acc_g;
                u[d] = acc_u;
            }
            for (int o = 0; o < MOE2_D_MODEL; o++) {
                float acc = 0;
                const float *dw = down + (size_t)o * MOE2_D_FF;
                for (int d = 0; d < MOE2_D_FF; d++)
                    acc += dw[d] * (u[d] * (g[d] > 0 ? 1.0f : 0.0f));
                contrib[o] = acc;
            }
            for (int o = 0; o < MOE2_D_MODEL; o++)
                out[o] += w[k] * contrib[o];
        }
    }
    return 0;
}

int wubu_moe2_init(wubu_moe2_t *moe, uint32_t seed)
{
    if (!moe) return -1;
    memset(moe, 0, sizeof(*moe));
    moe->router_w = (float *)malloc(MOE2_N_EXPERTS * MOE2_D_MODEL * sizeof(float));
    moe->shared_gate = (float *)malloc(MOE2_D_FF * MOE2_D_MODEL * sizeof(float));
    moe->shared_up = (float *)malloc(MOE2_D_FF * MOE2_D_MODEL * sizeof(float));
    moe->shared_down = (float *)malloc(MOE2_D_MODEL * MOE2_D_FF * sizeof(float));
    moe->shared_gate_w = (float *)malloc(MOE2_D_MODEL * sizeof(float));
    for (int e = 0; e < MOE2_N_EXPERTS; e++) {
        moe->exp_gate[e] = (float *)malloc(MOE2_D_FF * MOE2_D_MODEL * sizeof(float));
        moe->exp_up[e] = (float *)malloc(MOE2_D_FF * MOE2_D_MODEL * sizeof(float));
        moe->exp_down[e] = (float *)malloc(MOE2_D_MODEL * MOE2_D_FF * sizeof(float));
    }
    uint32_t rng = seed ? seed : 1;
    float scale = 0.05f;
    for (size_t i = 0; i < (size_t)MOE2_N_EXPERTS * MOE2_D_MODEL; i++) {
        rng = rng * 1103515245u + 12345u;
        moe->router_w[i] = scale * ((float)(rng >> 8) / 8388608.0f - 1.0f);
    }
    for (size_t i = 0; i < (size_t)MOE2_D_FF * MOE2_D_MODEL; i++) {
        rng = rng * 1103515245u + 12345u;
        moe->shared_gate[i] = scale * ((float)(rng >> 8) / 8388608.0f - 1.0f);
        moe->shared_up[i] = scale * ((float)(rng >> 8) / 8388608.0f - 1.0f);
    }
    for (size_t i = 0; i < (size_t)MOE2_D_MODEL * MOE2_D_FF; i++) {
        rng = rng * 1103515245u + 12345u;
        moe->shared_down[i] = scale * ((float)(rng >> 8) / 8388608.0f - 1.0f);
    }
    for (size_t i = 0; i < MOE2_D_MODEL; i++) {
        rng = rng * 1103515245u + 12345u;
        moe->shared_gate_w[i] = scale * ((float)(rng >> 8) / 8388608.0f - 1.0f);
    }
    for (int e = 0; e < MOE2_N_EXPERTS; e++)
        for (size_t i = 0; i < (size_t)MOE2_D_FF * MOE2_D_MODEL; i++) {
            rng = rng * 1103515245u + 12345u;
            moe->exp_gate[e][i] = scale * ((float)(rng >> 8) / 8388608.0f - 1.0f);
            moe->exp_up[e][i] = scale * ((float)(rng >> 8) / 8388608.0f - 1.0f);
        }
    for (int e = 0; e < MOE2_N_EXPERTS; e++)
        for (size_t i = 0; i < (size_t)MOE2_D_MODEL * MOE2_D_FF; i++) {
            rng = rng * 1103515245u + 12345u;
            moe->exp_down[e][i] = scale * ((float)(rng >> 8) / 8388608.0f - 1.0f);
        }
    return 0;
}

void wubu_moe2_free(wubu_moe2_t *moe)
{
    if (!moe) return;
    free(moe->router_w);
    free(moe->shared_gate); free(moe->shared_up); free(moe->shared_down);
    free(moe->shared_gate_w);
    for (int e = 0; e < MOE2_N_EXPERTS; e++) {
        free(moe->exp_gate[e]); free(moe->exp_up[e]); free(moe->exp_down[e]);
    }
    memset(moe, 0, sizeof(*moe));
}
