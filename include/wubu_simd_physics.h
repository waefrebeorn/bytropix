#ifndef WUBU_SIMD_PHYSICS_H
#define WUBU_SIMD_PHYSICS_H
/* wubu_simd_physics.h — WORLDSIM #607: vectorized physics (AVX2).
 *
 * The batched velocity/position integration kernel. Bodies are stored in
 * SoA (struct-of-arrays: separate dx/dy/dz arrays) so 8 bodies integrate
 * per AVX2 register. Two paths over the SAME data:
 *   - wubu_simd_integrate_scalar(): the reference C loop.
 *   - wubu_simd_integrate_avx2():   the AVX2 path (8-wide).
 * Both apply: drag (v *= 1-drag), gravity (dy += g*dt), integrate
 * (pos += v*dt). The benchmark test_avx2_physics proves the AVX2 path is
 * faster than scalar on a measured body count and produces identical
 * results (within float tolerance).
 *
 * C11; the AVX2 path compiles only when the target supports it (guarded
 * by __AVX2__); otherwise it falls back to the scalar loop, so the build
 * never breaks on a non-AVX2 host.
 */
#include <stddef.h>

/* Integrate n bodies (SoA arrays, n arbitrary). Scalar reference. */
void wubu_simd_integrate_scalar(const float *dx, const float *dy, const float *dz,
                                float *px, float *py, float *pz,
                                int n, float drag, float g, float dt);

/* AVX2 path (8-wide); identical semantics, falls back to scalar when
 * __AVX2__ is unavailable. */
void wubu_simd_integrate_avx2(const float *dx, const float *dy, const float *dz,
                              float *px, float *py, float *pz,
                              int n, float drag, float g, float dt);

/* Returns 1 if this build has a real AVX2 path (__AVX2__ defined). */
int wubu_simd_has_avx2(void);

#endif
