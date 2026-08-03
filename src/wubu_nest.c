/*
 * wubu_nest.c -- the WuBu Nesting transitions (層疊嵌套), phase 3.
 */
#include "wubu_nest.h"
#include <math.h>
#include <string.h>

wubu_quat_t wubu_quat_mul(wubu_quat_t a, wubu_quat_t b)
{
    wubu_quat_t r;
    r.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    r.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    r.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    r.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    return r;
}

wubu_quat_t wubu_quat_normalize(wubu_quat_t q)
{
    float n = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (n < 1e-12f) { q.w = 1; q.x = q.y = q.z = 0; return q; }
    q.w /= n; q.x /= n; q.y /= n; q.z /= n;
    return q;
}

void wubu_quat_rotate_vec(wubu_quat_t q, const float v[4], float out[4])
{
    /* out = q ⊗ v ⊗ q*  (the Hamilton double-cover rotation) */
    wubu_quat_t vq = { 0, v[0], v[1], v[2] };
    wubu_quat_t qc = { q.w, -q.x, -q.y, -q.z };
    wubu_quat_t mid = wubu_quat_mul(q, vq);
    wubu_quat_t r = wubu_quat_mul(mid, qc);
    out[0] = r.x; out[1] = r.y; out[2] = r.z;
    /* the 4th component is preserved by the rotation (SO(4) acts on
     * the full 4-vector; we use the scalar part as the 4th dim) */
    out[3] = v[3] + 0.0f * r.w;   /* SO(4): |v| preserved */
}

wubu_quat_t wubu_nest_learned_rotation(const float ld[4], float angle)
{
    /* the axis is the normalized descriptor; the angle is learned.
     * q = (cos(a/2), sin(a/2) * axis) */
    float ax = ld[0], ay = ld[1], az = ld[2];
    float an = sqrtf(ax * ax + ay * ay + az * az);
    if (an < 1e-12f) { an = 1; ax = 0; ay = 0; az = 1; }
    float s = sinf(angle * 0.5f) / an;
    wubu_quat_t q;
    q.w = cosf(angle * 0.5f);
    q.x = ax * s; q.y = ay * s; q.z = az * s;
    return q;
}

void wubu_nest_transition(wubu_quat_t rot, const float *v_src, int n_src,
                          const float *map_w, const float *map_b,
                          int n_dst, float *v_dst)
{
    /* rotate the source (if n_src >= 4; else pad) */
    float vr[8], v4[4] = { 0, 0, 0, 0 };
    for (int i = 0; i < n_src && i < 4; i++) v4[i] = v_src[i];
    wubu_quat_rotate_vec(rot, v4, v4);
    for (int i = 0; i < n_src && i < 4; i++) vr[i] = v4[i];
    for (int i = 4; i < n_src; i++) vr[i] = v_src[i];
    /* the non-rotational map: T̃(v) = tanh(W v + b) */
    for (int o = 0; o < n_dst; o++) {
        float acc = map_b ? map_b[o] : 0;
        const float *row = map_w + (size_t)o * n_src;
        for (int i = 0; i < n_src; i++) acc += row[i] * vr[i];
        v_dst[o] = tanhf(acc);
    }
}

void wubu_nest_relative(const float *v, const float *v_boundary,
                        int n, float *d)
{
    for (int i = 0; i < n; i++) d[i] = v[i] - v_boundary[i];
}

void wubu_nest_descriptor_flow(wubu_quat_t rot, const float *ld_src, int n,
                               const float *map_w, const float *map_b,
                               float sigma, int n_dst, float *ld_dst)
{
    /* the descriptor rotates with the data, maps, then σ_i is appended
     * as the last context dimension (the spread context pass) */
    float v4[4] = { 0, 0, 0, 0 }, vr[8];
    for (int i = 0; i < n && i < 4; i++) v4[i] = ld_src[i];
    wubu_quat_rotate_vec(rot, v4, v4);
    for (int i = 0; i < n && i < 4; i++) vr[i] = v4[i];
    for (int i = 4; i < n; i++) vr[i] = ld_src[i];
    for (int o = 0; o < n_dst - 1; o++) {
        float acc = map_b ? map_b[o] : 0;
        const float *row = map_w + (size_t)o * n;
        for (int i = 0; i < n; i++) acc += row[i] * vr[i];
        ld_dst[o] = tanhf(acc);
    }
    if (n_dst > 0) ld_dst[n_dst - 1] = sigma;
}
