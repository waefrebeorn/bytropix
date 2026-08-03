/*
 * wubu_foldmath.h -- the folded sin/cos, Silas Lock's algorithm as
 * presented in Kaze Emanuar's "The Folded Polynomial" (N64, 2023):
 * define ONE small polynomial on the first quarter [0, pi/4] and cover
 * the whole circle with the sine-wave symmetries (the quadrant fold).
 * On the N64 the sqrt gave the missing value cheaply; on Zen 4 the
 * odd/even polynomial PAIR is cheaper than the sqrt, so this version
 * computes both sin(r) and cos(r) as tiny polynomials -- branchless,
 * no sqrt, no table, no libm: the "compute" arm of the compute-vs-
 * fetch roofline (the N64 sine-table mis-optimization lesson).
 *
 * HEADER-ONLY (static inline): the fold must be INLINE in the caller
 * so the vectorizer sees the whole thing -- an opaque call cannot be
 * SIMD'd, and libm's loop rides __svml_sinf8. Compiled inline, the
 * fold vectorizes to the same width and beats the sinf/cosf loop.
 *
 * The fold: x = n*pi/2 + r, n = round(x*2/pi), r in [-pi/4, pi/4].
 *   sin(r) = r*P(r^2)  (odd, Taylor to r^9: err < 5e-7)
 *   cos(r) = Q(r^2)    (even, Taylor to r^8: err < 1.2e-6)
 *   q = n mod 4 (branchless bitwise selects):
 *     q0: s=+sin r   c=+cos r      q2: s=-sin r   c=-cos r
 *     q1: s=+cos r   c=-sin r      q3: s=-cos r   c=+sin r
 * Float Cody-Waite reduction (accurate for |x| < ~1e4 -- the RoPE +
 * game ranges; degrades gracefully beyond).
 */
#ifndef WUBU_FOLDMATH_H
#define WUBU_FOLDMATH_H

#include <math.h>

/* the Taylor coefficients for sin on [-pi/4, pi/4] (odd in r) */
#define FMS1  1.0f
#define FMS3 -0.166666666666666666f
#define FMS5  0.008333333333333333f
#define FMS7 -0.000198412698412698f
#define FMS9  0.000002755731922399f

/* the Taylor coefficients for cos (even in r) */
#define FMC2 -0.5f
#define FMC4  0.041666666666666667f
#define FMC6 -0.001388888888888889f
#define FMC8  0.000024801587301587f

/* the Cody-Waite two-part pi/2 (float) */
#define FMPI2_HI 1.57079637050628662109375f
#define FMPI2_LO -4.37113900018624283e-8f
#define FMTWO_OVER_PI 0.6366197723675814f

static inline float wubu_fold_sin_poly(float r)
{
    float r2 = r * r;
    return r * (FMS1 + r2 * (FMS3 + r2 * (FMS5 + r2 * (FMS7 + r2 * FMS9))));
}

static inline float wubu_fold_cos_poly(float r)
{
    float r2 = r * r;
    return 1.0f + r2 * (FMC2 + r2 * (FMC4 + r2 * (FMC6 + r2 * FMC8)));
}

/* both values in one call: one reduction, two tiny polynomials, no
 * sqrt, no branches -- bitwise &/| selects (no short-circuit) so the
 * whole fold vectorizes to the SIMD width. */
static inline void wubu_fold_sincos(float x, float *s, float *c)
{
    float nf = roundf(x * FMTWO_OVER_PI);
    /* r = x - n*pi/2, Cody-Waite */
    float r = fmaf(-nf, FMPI2_HI, x);
    r = fmaf(-nf, FMPI2_LO, r);
    float vs = wubu_fold_sin_poly(r);   /* sin(r), signed */
    float vc = wubu_fold_cos_poly(r);   /* cos(r), even */
    /* the quadrant as a FLOAT in [0,4) -- no int conversions */
    float q = nf - 4.0f * floorf(nf * 0.25f);
    float odd  = ((q >= 1.0f) & (q < 2.0f)) | (q >= 3.0f) ? 1.0f : 0.0f;
    float s_mag = fmaf(odd, vc - vs, vs);                  /* odd ? vc : vs */
    float c_mag = fmaf(odd, vs - vc, vc);                  /* odd ? vs : vc */
    float s_neg = (q >= 2.0f) ? -1.0f : 1.0f;
    float c_neg = (q >= 1.0f) & (q < 3.0f) ? -1.0f : 1.0f;
    *s = s_mag * s_neg;
    *c = c_mag * c_neg;
}

static inline float wubu_fold_sin(float x)
{
    float s, c;
    wubu_fold_sincos(x, &s, &c);
    return s;
}

static inline float wubu_fold_cos(float x)
{
    float s, c;
    wubu_fold_sincos(x, &s, &c);
    return c;
}

#endif
