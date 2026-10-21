/*
 * wubu_quat.h -- quaternion rigid-body orientation, integration and
 * spherical/squad interpolation.
 *
 * GOALS-1000:
 *   #974 (WORLDSIM, P0): rigid-body orientation + torque integration
 *   #998 (WORLDSIM, P0): slerp/squad interpolation for entity pose
 *
 * Representation: (w, x, y, z) with w the SCALAR part (Hamilton product),
 * i.e. the Shoemack convention. Unit quaternions encode rotations; the
 * double cover q == -q is NOT sign-normalised by most ops so that callers
 * can preserve continuity across slerp segments (squad tangents honour it).
 *
 * All ops are self-contained (math.h only) so the unit test links -lm and
 * nothing else. The implementation mirrors the independent Python reference
 * quat_oracle.py function-for-function, so the float results agree with the
 * double-precision oracle to ~1e-7 (well inside the 1e-5 KAT tolerance).
 */
#ifndef WUBU_QUAT_H
#define WUBU_QUAT_H

#include <stddef.h>   /* size_t (kept for ABI stability; ops are POD) */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* POD value type -- intentionally complete in the header so tests can
 * build instances directly on the stack (no opaque-ctx indirection needed).
 * Scalar w first, then vector (x,y,z). */
typedef struct wubu_quat { float w, x, y, z; } wubu_quat;

/* ---- construction ------------------------------------------------------- */

/* (1, 0, 0, 0): identity rotation. */
wubu_quat wubu_quat_identity(void);

/* Rotation of `angle_rad` about the (need not be unit) axis. Zero axis or
 * zero length collapses to identity. */
wubu_quat wubu_quat_from_axis_angle(const float axis[3], float angle_rad);

/* ---- algebra ------------------------------------------------------------ */

/* |q| (Euclidean norm of the 4-vector). */
float wubu_quat_norm(wubu_quat q);
/* 4-D inner product. */
float wubu_quat_dot(wubu_quat a, wubu_quat b);
/* Unit quaternion; a zero-norm input yields the identity (no NaN). */
wubu_quat wubu_quat_normalize(wubu_quat q);
/* Conjugate (w, -x, -y, -z). */
wubu_quat wubu_quat_conjugate(wubu_quat q);
/* General inverse q^-1 = conj(q)/|q|^2; zero-norm collapses to identity. */
wubu_quat wubu_quat_inverse(wubu_quat q);
/* Componentwise a + b (used by squad tangent accumulation). */
wubu_quat wubu_quat_add(wubu_quat a, wubu_quat b);
/* Hamilton product a * b (NOT commutative: a before b in the rotation). */
wubu_quat wubu_quat_mul(wubu_quat a, wubu_quat b);

/* Rotate the 3-vector v by the UNIT quaternion q into out[3].
 * out = q * (0,v) * conj(q). */
void wubu_quat_rotate(wubu_quat q, const float v[3], float out[3]);

/* ---- manifold log / exp (for integration + squad tangents) ------------- */

/* Lie-algebra log of a unit quaternion -> pure-vector quaternion
 * (0, theta * axis), theta = 2*atan2(|v|, w). */
wubu_quat wubu_quat_log(wubu_quat q);
/* Matrix-exp of a (possibly non-unit) quaternion; exp((0,v)) is the
 * axis-angle rotation, exp((s,v)) = e^s * exp((0,v)). */
wubu_quat wubu_quat_exp(wubu_quat q);

/* ---- interpolation ------------------------------------------------------ */

/* Spherical linear interpolation, shortest path. For near-parallel inputs
 * (|dot| > 0.9995) falls back to normalised lerp to avoid divide-by-zero.
 * t in [0,1]; endpoints are exact (returns a at t=0, b at t=1). */
wubu_quat wubu_quat_slerp(wubu_quat a, wubu_quat b, float t);

/* Shoemake squad control points for the knot sequence p0..p3.
 * out_s1 is the tangent at p1, out_s2 the tangent at p2. */
void wubu_quat_squad_tangents(wubu_quat p0, wubu_quat p1,
                              wubu_quat p2, wubu_quat p3,
                              wubu_quat *out_s1, wubu_quat *out_s2);

/* Spherical quadrangle: C^1-continuous interpolation across the segment
 * p1->p2 using pre-computed tangents s1, s2. */
wubu_quat wubu_quat_squad(wubu_quat q1, wubu_quat s1, wubu_quat s2,
                          wubu_quat q2, float t);

/* ---- rigid-body integration (#974) ------------------------------------- */

/* Advance orientation q by angular velocity `omega` (rad/s) over `dt`:
 *   q' = q * exp(0.5 * dt * omega);  result re-normalised to unit length. */
wubu_quat wubu_quat_integrate(wubu_quat q, const float omega[3], float dt);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_QUAT_H */
