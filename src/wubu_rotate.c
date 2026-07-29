/*
 * wubu_rotate.c -- Hadamard rotation for outlier suppression (doc 013).
 * Self-contained C11. See header for the invariance proof.
 */
#include "wubu_rotate.h"
#include <math.h>
#include <string.h>

int wubu_pow2_floor(int n) {
    int p = 1;
    while (p <= n/2 && p < (1<<30)) p <<= 1;
    return p;
}

/* In-place normalized Walsh-Hadamard of v[0..n), n power of 2.
 * Each of the log2(n) butterfly stages multiplies by 1/sqrt(2), for a
 * total 1/sqrt(n) normalization so H * H^T = I (H symmetric -> H*H = I). */
void wubu_hadamard(float *v, int n) {
    float stage_scale = 1.0f / sqrtf(2.0f);
    for (int step = n >> 1; step > 0; step >>= 1) {
        for (int i = 0; i < n; i += 2*step) {
            for (int j = i; j < i+step; j++) {
                float a = v[j];
                float b = v[j+step];
                v[j]       = (a + b) * stage_scale;
                v[j+step] = (a - b) * stage_scale;
            }
        }
    }
}

/* Fuse H_P (P = pow2 <= cols) on the RIGHT of W[rows x cols]:
 * store  W[r, c] <- sum_{j=0..P-1} W[r, j] * H_P[j, c]   for c < P.
 * Tail cols c>=P unchanged. After this, quantize W; at inference
 * rotate the input x prefix by wubu_rotate_input(x, P) before the GEMV. */
void wubu_rotate_fuse_right(float *W, int rows, int cols) {
    int P = wubu_pow2_floor(cols);
    if (P <= 1) return;
    /* H_P applied to cols: each output col c = sum_j W[r,j]*H[j,c].
     * H_P[j,c] = (1/sqrt(P)) * (-1)^{popcount(j & c)} for normalized.
     * Compute via the fast Hadamard on a copy of the P-wide col slice. */
    float *tmp = (float *)malloc((size_t)P * sizeof(float));
    if (!tmp) return;
    for (int r = 0; r < rows; r++) {
        float *wr = W + (size_t)r * cols;
        memcpy(tmp, wr, (size_t)P * sizeof(float));
        wubu_hadamard(tmp, P);
        memcpy(wr, tmp, (size_t)P * sizeof(float));
    }
    free(tmp);
}

/* Rotate the first P = pow2 <= n entries of x in place (online step).
 * Returns P so the caller knows how many leading dims were rotated
 * (must match the P used in wubu_rotate_fuse_right). */
int wubu_rotate_input(float *x, int n) {
    int P = wubu_pow2_floor(n);
    if (P <= 1) return 1;
    wubu_hadamard(x, P);
    return P;
}
