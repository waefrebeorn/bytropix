/*
 * wubu_hadamard.h — Hadamard rotation for incoherent FP8 (doc H03, CPU core).
 *
 * Incoherent FP8 (e.g. QuaRot / SpinQuant) rotates weight and activation
 * tensors by a random orthogonal (Hadamard) matrix so that magnitude outliers
 * are spread across channels. This makes the downstream FP8/B07 quantization
 * much lower-error. The Hadamard transform itself is pure linear algebra — no
 * GPU required. We provide fast Walsh-Hadamard (powers of two) and a general
 * recursive transform, plus the "rotate weight row" helper used before FP8
 * packing. Ties to B07 (FP8) and 013 (QuaRot Hadamard fuse).
 */
#ifndef WUBU_HADAMARD_H
#define WUBU_HADAMARD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* In-place Walsh-Hadamard transform of x[0..n) where n is a power of two.
 * Normalized by 1/sqrt(n) so it is orthonormal (H * H^T = I). */
void wubu_hadamard_fwd(float *x, int n);
void wubu_hadamard_inv(float *x, int n);  /* same as fwd (self-inverse up to norm) */

/* Rotate a [rows*d] matrix M in place by H on the right (M <- M * H / sqrt(d)).
 * Used to rotate weight rows before FP8 packing (incoherence). */
void wubu_hadamard_rotate_rows(float *M, int rows, int d);

/* Rotate a [d] activation vector by H (v <- H * v / sqrt(d)). */
void wubu_hadamard_rotate_vec(float *v, int d);

/* Whether n is a power of two (Walsh-Hadamard fast path valid). */
int wubu_is_pow2(int n);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_HADAMARD_H */
