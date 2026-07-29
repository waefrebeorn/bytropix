/*
 * wubu_rotate.h -- Hadamard (Walsh-Hadamard) rotation for outlier
 * suppression before quantization (QuaRot / SpinQuant convergent idea, doc 013).
 *
 * Invariant (computational invariance): for an orthonormal H,
 *     y = (W * H) * (H * x) = W * H * H * x = W * x
 * so the network OUTPUT is unchanged; only the *distribution* of weights
 * and activations is decorrelated, which lets a uniform int4/int8 quantizer
 * keep its full dynamic range instead of wasting it on a few outlier dims.
 *
 * We rotate the pow2-aligned PREFIX of each dim (P = largest pow2 <= n),
 * leaving the tail unrotated -- still exactly invariant overall.
 */
#ifndef WUBU_ROTATE_H
#define WUBU_ROTATE_H

#include <stddef.h>
#include <stdint.h>

/* In-place Walsh-Hadamard transform of v[0..n), n MUST be a power of 2.
 * O(n log n), uses the iterative butterfly. Result is scaled by 1/sqrt(n)
 * (normalized Hadamard) so H * H^T = I. */
void wubu_hadamard(float *v, int n);

/* Largest power of 2 <= n (n >= 1). */
int wubu_pow2_floor(int n);

/* Fuse H on the INPUT side of a [rows x cols] weight:  W <- W * H_cols_prefix.
 * Each row r:  W[r,0..P) <- H_P * W[r,0..P)   (P = pow2 <= cols).
 * Tail cols P..cols-1 left unrotated. After this, quantize W normally;
 * at inference rotate the matching x prefix by wubu_hadamard(x, P). */
void wubu_rotate_fuse_right(float *W, int rows, int cols);

/* Rotate the first P = pow2 <= n entries of x in place (the online step
 * that pairs with wubu_rotate_fuse_right). Returns P. */
int wubu_rotate_input(float *x, int n);

#endif /* WUBU_ROTATE_H */
