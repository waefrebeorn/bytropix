/*
 * wubu_quat.c -- quaternion orientation, integration, slerp/squad.
 *
 * Implementation note: this file mirrors docs/scratch/quat_oracle.py
 * function-for-function (same formulae, same branch thresholds, same
 * zero-collapse semantics) so the C-float output and the Python-double
 * oracle agree to ~1e-7. See wubu_quat.h for the GOALS-1000 mapping.
 *
 * Magic constants are derived here, not imported:
 *   - 0.9995   : Shoemake's "near-parallel -> nlerp" knee (sin(theta)~theta,
 *                sin(theta) denominator would lose ~3 digits of precision).
 *   - 1e-12    : below this norm the quaternion carries no angle to extract
 *                (log/exp/divide collapse cleanly to the identity).
 */
#include "wubu_quat.h"
#include <math.h>

/* ---- construction ------------------------------------------------------- */

wubu_quat wubu_quat_identity(void)
{
    return (wubu_quat){ 1.0f, 0.0f, 0.0f, 0.0f };
}

wubu_quat wubu_quat_from_axis_angle(const float axis[3], float angle_rad)
{
    float nx = axis[0], ny = axis[1], nz = axis[2];
    float n = sqrtf(nx * nx + ny * ny + nz * nz);
    if (n < 1e-12f)                 /* degenerate axis: no rotation */
        return wubu_quat_identity();
    float s = 1.0f / n;             /* normalise axis */
    float ax = nx * s, ay = ny * s, az = nz * s;
    float h = 0.5f * angle_rad;
    float sh = sinf(h), ch = cosf(h);
    return (wubu_quat){ ch, ax * sh, ay * sh, az * sh };
}

/* ---- algebra ------------------------------------------------------------ */

float wubu_quat_norm(wubu_quat q)
{
    return sqrtf(wubu_quat_dot(q, q));
}

float wubu_quat_dot(wubu_quat a, wubu_quat b)
{
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

wubu_quat wubu_quat_normalize(wubu_quat q)
{
    float n = wubu_quat_norm(q);
    if (n < 1e-12f)                 /* zero quaternion -> identity, no NaN */
        return wubu_quat_identity();
    float s = 1.0f / n;
    return (wubu_quat){ q.w * s, q.x * s, q.y * s, q.z * s };
}

wubu_quat wubu_quat_conjugate(wubu_quat q)
{
    return (wubu_quat){ q.w, -q.x, -q.y, -q.z };
}

wubu_quat wubu_quat_inverse(wubu_quat q)
{
    float nn = wubu_quat_dot(q, q);
    if (nn < 1e-24f)                /* |q|^2 ~ 0: collapse to identity */
        return wubu_quat_identity();
    float s = 1.0f / nn;
    return (wubu_quat){ q.w * s, -q.x * s, -q.y * s, -q.z * s };
}

wubu_quat wubu_quat_add(wubu_quat a, wubu_quat b)
{
    return (wubu_quat){ a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z };
}

wubu_quat wubu_quat_mul(wubu_quat a, wubu_quat b)
{
    float aw = a.w, ax = a.x, ay = a.y, az = a.z;
    float bw = b.w, bx = b.x, by = b.y, bz = b.z;
    return (wubu_quat){
        aw * bw - ax * bx - ay * by - az * bz,     /* w */
        aw * bx + ax * bw + ay * bz - az * by,     /* x */
        aw * by - ax * bz + ay * bw + az * bx,     /* y */
        aw * bz + ax * by - ay * bx + az * bw      /* z */
    };
}

void wubu_quat_rotate(wubu_quat q, const float v[3], float out[3])
{
    /* v' = q * (0,v) * conj(q). For a unit q this is the canonical rotate. */
    wubu_quat vq = { 0.0f, v[0], v[1], v[2] };
    wubu_quat r = wubu_quat_mul(wubu_quat_mul(q, vq), wubu_quat_conjugate(q));
    out[0] = r.x;
    out[1] = r.y;
    out[2] = r.z;
}

/* ---- manifold log / exp ------------------------------------------------- */

wubu_quat wubu_quat_log(wubu_quat q)
{
    float vx = q.x, vy = q.y, vz = q.z;
    float vn = sqrtf(vx * vx + vy * vy + vz * vz);
    if (vn < 1e-12f)
        return (wubu_quat){ 0.0f, 0.0f, 0.0f, 0.0f };   /* on the scalar axis */
    float a = atan2f(vn, q.w);
    float s = a / vn;
    return (wubu_quat){ 0.0f, vx * s, vy * s, vz * s };
}

wubu_quat wubu_quat_exp(wubu_quat q)
{
    float vx = q.x, vy = q.y, vz = q.z;
    float vn = sqrtf(vx * vx + vy * vy + vz * vz);
    float ew = expf(q.w);
    if (vn < 1e-12f)
        return (wubu_quat){ ew, 0.0f, 0.0f, 0.0f };
    float s = sinf(vn) / vn;
    return (wubu_quat){ ew * cosf(vn), ew * s * vx, ew * s * vy, ew * s * vz };
}

/* ---- interpolation ------------------------------------------------------ */

/* Shoemake 1985 shortest-arc slerp. Mirrors quat_oracle.slerp exactly:
 *   n = |a||b|;  d = <a,b>/n;  if d<0 flip b (shortest path);
 *   if d>0.9995 -> nlerp (avoid sin denominator collapse);
 *   else SLERP via acos(d). */
wubu_quat wubu_quat_slerp(wubu_quat a, wubu_quat b, float t)
{
    float n = wubu_quat_norm(a) * wubu_quat_norm(b);
    if (n < 1e-12f)
        return a;                   /* degenerate: keep a as-is */
    float d = wubu_quat_dot(a, b) / n;
    if (d < 0.0f) {                 /* take the shorter arc */
        b.w = -b.w; b.x = -b.x; b.y = -b.y; b.z = -b.z;
        d = -d;
    }
    if (d > 0.9995f) {              /* near-parallel: nlerp fallback */
        wubu_quat r = {
            a.w * (1.0f - t) + b.w * t,
            a.x * (1.0f - t) + b.x * t,
            a.y * (1.0f - t) + b.y * t,
            a.z * (1.0f - t) + b.z * t
        };
        return wubu_quat_normalize(r);
    }
    float th = acosf(d);
    float s = sinf(th);
    float wa = sinf((1.0f - t) * th) / s;
    float wb = sinf(t * th) / s;
    return (wubu_quat){
        wa * a.w + wb * b.w,
        wa * a.x + wb * b.x,
        wa * a.y + wb * b.y,
        wa * a.z + wb * b.z
    };
}

void wubu_quat_squad_tangents(wubu_quat p0, wubu_quat p1,
                              wubu_quat p2, wubu_quat p3,
                              wubu_quat *out_s1, wubu_quat *out_s2)
{
    /* s_i = p_i * exp( -1/4 * (log(p_i^-1 * p_{i-1}) + log(p_i^-1 * p_{i+1})) ) */
    wubu_quat inv_p1 = wubu_quat_inverse(p1);
    wubu_quat inv_p2 = wubu_quat_inverse(p2);
    wubu_quat l1 = wubu_quat_add(wubu_quat_log(wubu_quat_mul(inv_p1, p0)),
                                 wubu_quat_log(wubu_quat_mul(inv_p1, p2)));
    wubu_quat l2 = wubu_quat_add(wubu_quat_log(wubu_quat_mul(inv_p2, p1)),
                                 wubu_quat_log(wubu_quat_mul(inv_p2, p3)));
    wubu_quat e1 = wubu_quat_exp((wubu_quat){ -0.25f * l1.w, -0.25f * l1.x,
                                              -0.25f * l1.y, -0.25f * l1.z });
    wubu_quat e2 = wubu_quat_exp((wubu_quat){ -0.25f * l2.w, -0.25f * l2.x,
                                              -0.25f * l2.y, -0.25f * l2.z });
    if (out_s1) *out_s1 = wubu_quat_mul(p1, e1);
    if (out_s2) *out_s2 = wubu_quat_mul(p2, e2);
}

wubu_quat wubu_quat_squad(wubu_quat q1, wubu_quat s1, wubu_quat s2,
                          wubu_quat q2, float t)
{
    /* S(q1,s1,s2,q2;t) = slerp(slerp(q1,q2,t), slerp(s1,s2,t), 2t(1-t)) */
    return wubu_quat_slerp(wubu_quat_slerp(q1, q2, t),
                           wubu_quat_slerp(s1, s2, t),
                           2.0f * t * (1.0f - t));
}

/* ---- rigid-body integration (#974) ------------------------------------- */

wubu_quat wubu_quat_integrate(wubu_quat q, const float omega[3], float dt)
{
    /* q' = q * exp( (dt/2) * omega )  -- omega as a pure-vector quaternion. */
    float half = 0.5f * dt;
    wubu_quat w = { 0.0f, half * omega[0], half * omega[1], half * omega[2] };
    return wubu_quat_normalize(wubu_quat_mul(q, wubu_quat_exp(w)));
}
