/*
 * wubu_nest.h -- the WuBu Nesting transitions (層疊嵌套), phase 3.
 *
 * The nesting paper (THEORY/03-wubu-nesting-paper.md) specifies the
 * inter-level transition:
 *
 *     T_{i→i+1} = T̃_i ∘ R_i
 *
 * where R_i is a quaternion SO(4) rotation applied in the tangent
 * space of the source hyperbolic bubble, and T̃_i is a non-rotational
 * mapping (dimension change + nonlinearity) into the target tangent
 * space. The rotation is applied SIMULTANEOUSLY to the primary
 * representation, the boundary sub-manifold vectors, and the level
 * descriptor ld_i -- so relative orientations survive the transition.
 *
 * This module implements the transition math: quaternion multiplication
 * (the Hamilton product), the SO(4) double-cover rotation, the level
 * descriptor flow, and the spread parameter σ_i context pass.
 *
 * The quaternion formulas are the ones in MATH/ (Hamilton product,
 * quaternion rotation) -- the same engine the nesting paper uses.
 */
#ifndef WUBU_NEST_H
#define WUBU_NEST_H

#include <stdint.h>

/* a quaternion (w, x, y, z) */
typedef struct { float w, x, y, z; } wubu_quat_t;

/* N1: the Hamilton product (MATH: hamilton_product). */
wubu_quat_t wubu_quat_mul(wubu_quat_t a, wubu_quat_t b);

/* N2: rotate a 4-vector by a unit quaternion (SO(4) double cover). */
void wubu_quat_rotate_vec(wubu_quat_t q, const float v[4], float out[4]);

/* N3: normalize a quaternion (unit quats only rotate). */
wubu_quat_t wubu_quat_normalize(wubu_quat_t q);

/* N4: a learned rotation interpolated from the source representation:
 * R_i = slerp(identity, target) with a learned angle. The rotation
 * axis is the normalized source descriptor. */
wubu_quat_t wubu_nest_learned_rotation(const float ld[4], float angle);

/* N5: the full inter-level transition in tangent space:
 *   v' = T̃_i(R_i(v)) -- rotate then map.
 * The non-rotational map is a learned affine + tanh (dimension
 * n_src -> n_dst). */
void wubu_nest_transition(wubu_quat_t rot, const float *v_src, int n_src,
                          const float *map_w,   /* [n_dst, n_src] */
                          const float *map_b,   /* [n_dst] */
                          int n_dst, float *v_dst);

/* N6: relative vectors between the primary and the boundary manifolds
 * after the rotation: d = v' - v'_boundary (rotation-aware hierarchy
 * relationships at the target scale). */
void wubu_nest_relative(const float *v, const float *v_boundary,
                        int n, float *d);

/* N7: the level descriptor flow: ld_{i+1} = T̃_i(R_i(ld_i)) with the
 * spread σ_i passed as context (appended to the descriptor). */
void wubu_nest_descriptor_flow(wubu_quat_t rot, const float *ld_src, int n,
                               const float *map_w, const float *map_b,
                               float sigma, int n_dst, float *ld_dst);

#endif
