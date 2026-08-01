/*
 * wubu_hadamard.c — Hadamard rotation for incoherent FP8 (doc H03). C11.
 */
#include "wubu_hadamard.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int wubu_is_pow2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/* Recursive Walsh-Hadamard transform in-place (unnormalized). */
static void wht_rec(float *x, int n) {
    if (n <= 1) return;
    int h = n / 2;
    float *t = (float *)malloc(n * sizeof(float));
    if (!t) return;
    for (int i = 0; i < h; i++) {
        float s = x[i], y = x[i + h];
        t[i]     = s + y;
        t[i + h] = s - y;
    }
    memcpy(x, t, n * sizeof(float));
    free(t);
    wht_rec(x, h);
    wht_rec(x + h, h);
}

void wubu_hadamard_fwd(float *x, int n) {
    if (!wubu_is_pow2(n) || !x) return;
    wht_rec(x, n);
    float norm = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) x[i] *= norm;
}

void wubu_hadamard_inv(float *x, int n) {
    /* Walsh-Hadamard is orthogonal & symmetric; fwd == inverse up to 1/n, and
     * our normalized fwd already divides by sqrt(n), so applying twice gives
     * the identity. Hence inverse == fwd. */
    wubu_hadamard_fwd(x, n);
}

void wubu_hadamard_rotate_vec(float *v, int d) {
    wubu_hadamard_fwd(v, d);
}

void wubu_hadamard_rotate_rows(float *M, int rows, int d) {
    for (int r = 0; r < rows; r++)
        wubu_hadamard_fwd(M + (size_t)r * d, d);
}
