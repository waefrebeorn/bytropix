/*
 * wubu_linattn.c -- linear-attention / SSM frontier (Theme IU). C11.
 */
#include "wubu_linattn.h"
#include <math.h>
#include <string.h>

static float dot(const float *a, const float *b, int d)
{
    float s = 0;
    for (int i = 0; i < d; i++) s += a[i] * b[i];
    return s;
}

int wubu_la_chunk(const float *k, const float *v, const float *decay,
                  int n, int chunk, float *state, int d)
{
    if (!k || !v || !decay || !state || chunk <= 0 || d <= 0) return -1;
    int n_chunks = 0;
    for (int c = 0; c < n; c += chunk) {
        int end = c + chunk < n ? c + chunk : n;
        /* decay the state by the chunk's accumulated decay */
        float dec = 1.0f;
        for (int t = c; t < end; t++) dec *= decay[t];
        for (int i = 0; i < d; i++) state[i] *= dec;
        /* accumulate the chunk's k-v outer contributions */
        for (int t = c; t < end; t++)
            for (int i = 0; i < d; i++)
                state[i] += k[t * d + i] * v[t];
        n_chunks++;
    }
    return n_chunks;
}

int wubu_la_selective(const float *x, const float *B, const float *C,
                      const float *A, float *state, float *out, int d)
{
    if (!x || !B || !C || !A || !state || !out) return -1;
    /* the selective SSM: state = A*state + B*x; out = C'state */
    float dt = 1.0f;   /* the caller supplies the discretized A/B/C */
    for (int i = 0; i < d; i++)
        state[i] = A[i] * state[i] + B[i] * x[i] * dt;
    *out = dot(C, state, d);
    return 0;
}

int wubu_la_delta(const float *B, const float *target, const float *C,
                  float gate, float *state, int d)
{
    if (!B || !target || !C || !state) return -1;
    float err = dot(C, state, d);
    for (int i = 0; i < d; i++)
        state[i] += gate * B[i] * (target[i] - err);
    return 0;
}

int wubu_la_slots(const float *x, const float **slots, int n_slots,
                  int d, float gate, float *out)
{
    if (!x || !slots || !out || n_slots <= 0) return -1;
    for (int i = 0; i < d; i++) out[i] = 0;
    float total = 0;
    for (int s = 0; s < n_slots; s++) {
        float a = dot(x, slots[s], d);
        out[0] += a * slots[s][0] * gate;
        total += a * a;
        for (int i = 1; i < d; i++) out[i] += a * slots[s][i] * gate;
    }
    return total > 0 ? 1 : 0;
}

int wubu_la_hgrn(const float *x, const float *g1, const float *g2,
                 float *state, int d, float *out)
{
    if (!x || !g1 || !g2 || !state || !out) return -1;
    for (int i = 0; i < d; i++) {
        state[i] = g1[i] * state[i] + g2[i] * x[i];
        out[i] = state[i];
    }
    return 0;
}

int wubu_la_tile(const float *k, const float *v, int n, int d,
                 int tile, float *state)
{
    if (!k || !v || !state || tile <= 0 || d <= 0) return -1;
    int n_tiles = 0;
    for (int c = 0; c < n; c += tile) {
        int end = c + tile < n ? c + tile : n;
        for (int t = c; t < end; t++)
            for (int i = 0; i < d; i++)
                state[i] += k[t * d + i] * v[t];
        n_tiles++;
    }
    return n_tiles;
}

float wubu_la_lightning(float state, float k, float v, float decay)
{
    return decay * state + k * v;
}

int wubu_la_householder(float *vec, int d, int steps)
{
    if (!vec || d <= 0 || steps <= 0) return -1;
    for (int s = 0; s < steps; s++) {
        /* the standard Householder reflector: reflect v onto ||v||*e1.
         * H = I - 2 u u'/(u'u) with u = v - ||v|| e1; Hv = ||v|| e1,
         * so the norm is preserved (a rotation-like accumulation). */
        float nrm = 0;
        for (int i = 0; i < d; i++) nrm += vec[i] * vec[i];
        nrm = sqrtf(nrm);
        if (nrm < 1e-9f) return 0;
        float u0 = vec[0] - nrm;
        float usq = u0 * u0;
        for (int i = 1; i < d; i++) usq += vec[i] * vec[i];
        float uv = nrm * nrm - nrm * vec[0];   /* u'v */
        float coef = 2.0f * uv / (usq + 1e-9f);
        vec[0] -= coef * u0;
        for (int i = 1; i < d; i++) vec[i] -= coef * vec[i];
    }
    return 0;
}

int wubu_la_hybrid_heads(const float *x, int heads, int d,
                         int n_attn, int n_ssm, float *out)
{
    if (!x || !out || heads <= 0) return -1;
    (void)n_attn; (void)n_ssm; (void)d;
    /* the mix: attention heads scale, SSM heads persist (the caller
     * owns the per-head state); here we return the head count */
    *out = 0;
    return heads;
}

int wubu_la_kv_free(const float *x, const float *A, float *state,
                    int d, float *out)
{
    if (!x || !A || !state || !out) return -1;
    for (int i = 0; i < d; i++) {
        state[i] = A[i] * state[i] + x[i];
        out[i] = state[i];
    }
    return 0;
}

float wubu_la_stable(float acc, float clamp)
{
    if (clamp <= 0) return acc;
    if (acc > clamp) return clamp;
    if (acc < -clamp) return -clamp;
    return acc;
}
